// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>
#include <set>

#include <ygm/comm.hpp>
#include <ygm/detail/ygm_ptr.hpp>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename Point>
class query_engine;

template <typename DistType, typename Point>
class dhnsw_impl {
  friend class query_engine<DistType, Point>;

 public:
  using feature_vec_type         = Point;
  using data_index_cell_map_type = std::map<size_t, std::vector<size_t>>;

  dhnsw_impl(int max_voronoi_rank, int num_cells,
             hnswlib::SpaceInterface<DistType> *space_ptr, ygm::comm *c)
      : m_max_voronoi_rank(max_voronoi_rank),
        m_num_cells(num_cells),
        m_metric_space_ptr(space_ptr),
        m_comm_size(m_comm->size()),
        m_comm_rank(m_comm->rank()),
        m_comm(c),
        pthis(this) {
    size_t local_cells =
        m_num_cells / m_comm_size + (m_comm_rank < m_num_cells % m_comm_size);

    m_cell_add_vec.resize(local_cells);
  }

  ~dhnsw_impl() {
    for (int i = 0; i < m_voronoi_cell_hnsw.size(); ++i) {
      delete m_voronoi_cell_hnsw[i];
    }
  }

  void add_data_point_to_insertion_queue(const size_t index, const Point &v) {
    std::vector<size_t> closest_seeds;
    find_approx_closest_seeds(v, m_max_voronoi_rank, closest_seeds);
    add_data_point_to_insertion_queue(index, v, closest_seeds);
  }

  void add_data_point_to_insertion_queue(
      const size_t index, const Point &v,
      const std::vector<size_t> &closest_seeds) {
    auto insertion_cell = closest_seeds[0];
    m_comm->async(
        cell_owner(insertion_cell),
        [](auto mbox, ygm::ygm_ptr<dhnsw_impl> pthis, size_t index,
           const size_t               insertion_cell,
           const std::vector<size_t> &closest_seeds, const Point &v) {
          auto local_insertion_cell = pthis->local_cell_index(insertion_cell);
          pthis->try_add_closest_cells(index, closest_seeds);
          pthis->m_cell_add_vec[local_insertion_cell].push_back(
              std::make_pair(index, v));
        },
        pthis, index, insertion_cell, closest_seeds, v);
  }

  void initialize_hnsw() {
    // Initialize HNSW structures
    for (int i = 0; i < num_local_cells(); ++i) {
      hnswlib::HierarchicalNSW<float> *hnsw =
          new hnswlib::HierarchicalNSW<float>(
              m_metric_space_ptr, m_cell_add_vec[i].size(), 16, 200, 1);
      m_voronoi_cell_hnsw.push_back(hnsw);

      // Add data points to HNSW
      std::random_shuffle(m_cell_add_vec[i].begin(), m_cell_add_vec[i].end());
      for (auto &[index, feature_vec] : m_cell_add_vec[i]) {
        m_local_data[index] = std::move(feature_vec);
        m_voronoi_cell_hnsw[i]->addPoint(&m_local_data[index], index);
      }
      m_cell_add_vec[i].clear();
    }
    m_cell_add_vec.clear();
  }

  void flush_insertion_queues() {
    for (size_t i = 0; i < m_cell_add_vec.size(); ++i) {
      // Shuffle points to be added to each cell. Having points added
      // sequentially seems to give weird results sometimes...
      std::random_shuffle(m_cell_add_vec[i].begin(), m_cell_add_vec[i].end());
      for (auto &[index, feature_vec] : m_cell_add_vec[i]) {
        m_local_data[index] = feature_vec;
        m_voronoi_cell_hnsw[i]->addPoint(&m_local_data[index], index);
      }
      m_cell_add_vec[i].clear();
    }
    m_cell_add_vec.clear();
  }

  void fill_seed_hnsw() {
    m_seed_hnsw = new hnswlib::HierarchicalNSW<DistType>(
        m_metric_space_ptr, m_seeds.size(), 16, 200, 3149);

#pragma omp parallel for
    for (size_t i = 0; i < m_seeds.size(); ++i) {
      m_seed_hnsw->addPoint(&m_seeds[i], i);
    }
  }

  inline int cell_owner(size_t index) {
    return std::hash<size_t>{}(index) % m_comm_size;
  }

  inline int local_cell_index(size_t global_cell_index) const {
    return global_cell_index / m_comm_size;
  }

  inline int global_cell_index(size_t local_cell_index) const {
    return local_cell_index * m_comm_size + m_comm_rank;
  }

  inline int num_local_cells() {
    return m_num_cells / m_comm_size +
           (m_comm_rank < m_num_cells % m_comm_size);
  }

  void find_approx_closest_seeds(const Point         &sample_features,
                                 const int            num_closest_seeds,
                                 std::vector<size_t> &output) const {
    output.resize(num_closest_seeds);

    std::priority_queue<std::pair<float, hnswlib::labeltype>> nearest_seeds_pq =
        m_seed_hnsw->searchKnn(&sample_features, num_closest_seeds);

    size_t i = num_closest_seeds;
    while (nearest_seeds_pq.size() > 0) {
      auto seed_ID = nearest_seeds_pq.top().second;
      output[--i]  = seed_ID;
      nearest_seeds_pq.pop();
    }

    return;
  }

  void store_seeds(const std::vector<Point> &seed_features) {
    m_seeds.clear();

    for (size_t i = 0; i < seed_features.size(); ++i) {
      m_seeds.push_back(seed_features[i]);
    }
  }

  template <typename Function>
  void for_all_data(Function fn) {
    std::for_each(m_local_data.begin(), m_local_data.end(), fn);
  }

  inline ygm::comm &comm() { return *m_comm; }

  inline int max_voronoi_rank() { return m_max_voronoi_rank; }

  size_t global_size() {
    size_t hnsw_size{0};

    for (auto &voronoi_cell_hnsw : m_voronoi_cell_hnsw) {
      hnsw_size += voronoi_cell_hnsw->cur_element_count;
    }

    return comm().all_reduce_sum(hnsw_size);
  }

  const hnswlib::HierarchicalNSW<float> &get_cell_hnsw(const int cell) const {
    return *m_voronoi_cell_hnsw[local_cell_index(cell)];
  }

  const std::vector<size_t> &get_cell_pointers(size_t index) {
    return m_map_point_to_cells[index];
  }

 private:
  hnswlib::HierarchicalNSW<float> &get_cell_hnsw_non_const(
      const int cell) const {
    return *m_voronoi_cell_hnsw[local_cell_index(cell)];
  }

  void try_add_closest_cells(const size_t               index,
                             const std::vector<size_t> &cells) {
    for (const size_t &cell : cells) {
      try_add_closest_cell(index, cell);
    }
  }

  // Scans through full list of cells to try inserting.
  // Can do something better if necessary, but this list should be small.
  void try_add_closest_cell(const size_t index, const size_t cell) {
    for (const size_t &current_cell : get_cell_pointers(index)) {
      if (current_cell == cell) return;
    }
    m_map_point_to_cells[index].push_back(cell);
  }

  data_index_cell_map_type m_map_point_to_cells;

  std::map<size_t, Point> m_local_data;

  // Seeds
  std::vector<Point>                  m_seeds;
  hnswlib::HierarchicalNSW<DistType> *m_seed_hnsw;

  std::vector<hnswlib::HierarchicalNSW<DistType> *> m_voronoi_cell_hnsw;
  hnswlib::SpaceInterface<DistType>                *m_metric_space_ptr;

  std::vector<std::vector<std::pair<size_t, Point>>>
      m_cell_add_vec;  // per-cell vector of indices to add to HNSW structure

  ygm::comm               *m_comm;
  ygm::ygm_ptr<dhnsw_impl> pthis;
  int                      m_comm_size;
  int                      m_comm_rank;

  int m_max_voronoi_rank;
  int m_num_cells;
};

}  // namespace dhnsw_detail
}  // namespace saltatlas
