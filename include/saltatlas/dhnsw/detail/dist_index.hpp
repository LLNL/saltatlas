// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>

#include <ygm/comm.hpp>
#include <ygm/detail/ygm_ptr.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename IndexType, typename Point>
class query_engine;

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
class dhnsw_impl {
  friend class query_engine<DistType, IndexType, Point>;

 public:
  using dist_t                   = DistType;
  using index_t                  = IndexType;
  using point_t                  = Point;
  using feature_vec_type         = Point;
  using partitioner_t            = Partitioner<dist_t, index_t, point_t>;
  using index_vec_t              = std::vector<index_t>;
  using data_index_cell_map_type = std::map<index_t, index_vec_t>;

  dhnsw_impl(int max_voronoi_rank, int num_cells,
             hnswlib::SpaceInterface<dist_t> *space_ptr, ygm::comm *c,
             partitioner_t &p)
      : m_max_voronoi_rank(max_voronoi_rank),
        m_num_cells(num_cells),
        m_metric_space_ptr(space_ptr),
        m_comm_size(m_comm->size()),
        m_comm_rank(m_comm->rank()),
        m_comm(c),
        m_partitioner(p),
        pthis(this) {
    size_t local_cells =
        m_num_cells / m_comm_size + (m_comm_rank < m_num_cells % m_comm_size);

    m_cell_add_vec.resize(local_cells);
  }

  ~dhnsw_impl() {
    m_comm->barrier();
    for (int i = 0; i < m_voronoi_cell_hnsw.size(); ++i) {
      delete m_voronoi_cell_hnsw[i];
    }
  }

  template <typename Container>
  void partition_data(Container &data, const uint32_t num_partitions) {
    m_partitioner.initialize(data, num_partitions);
  }

  void add_data_point_to_insertion_queue(const index_t  index,
                                         const point_t &v) {
    index_vec_t point_partitions =
        partitioner().find_point_partitions(v, m_max_voronoi_rank);
    ASSERT_RELEASE(point_partitions[0] < m_num_cells);
    add_data_point_to_insertion_queue(index, v, point_partitions);
  }

  void add_data_point_to_insertion_queue(const index_t index, const point_t &v,
                                         const index_vec_t &closest_seeds) {
    auto insertion_cell = closest_seeds[0];
    ASSERT_RELEASE(insertion_cell < m_num_cells);
    m_comm->async(
        cell_owner(insertion_cell),
        [](auto mbox, ygm::ygm_ptr<dhnsw_impl> pthis, index_t index,
           const index_t insertion_cell, const index_vec_t &closest_seeds,
           const point_t &v) {
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
      ASSERT_RELEASE(m_cell_add_vec[i].size() > 0);

      hnswlib::HierarchicalNSW<dist_t> *hnsw =
          new hnswlib::HierarchicalNSW<dist_t>(
              m_metric_space_ptr, m_cell_add_vec[i].size(), 16, 200, 1);
      m_voronoi_cell_hnsw.push_back(hnsw);

      // Add data points to HNSW
      set_cell_add_vec_ordering(m_cell_add_vec[i]);
      for (auto &[index, feature_vec] : m_cell_add_vec[i]) {
        m_local_data[index] = std::move(feature_vec);
        m_voronoi_cell_hnsw[i]->addPoint(&m_local_data[index], index);
      }
      m_cell_add_vec[i].clear();
    }
    m_cell_add_vec.clear();
  }

  inline int cell_owner(index_t index) {
    return std::hash<index_t>{}(index) % m_comm_size;
  }

  inline int local_cell_index(index_t global_cell_index) const {
    return global_cell_index / m_comm_size;
  }

  inline int global_cell_index(index_t local_cell_index) const {
    return local_cell_index * m_comm_size + m_comm_rank;
  }

  inline int num_local_cells() {
    return m_num_cells / m_comm_size +
           (m_comm_rank < m_num_cells % m_comm_size);
  }

  inline ygm::comm &comm() { return *m_comm; }

  partitioner_t &partitioner() { return m_partitioner; }

  inline int max_voronoi_rank() { return m_max_voronoi_rank; }

  size_t global_size() {
    size_t hnsw_size{0};

    for (auto &voronoi_cell_hnsw : m_voronoi_cell_hnsw) {
      hnsw_size += voronoi_cell_hnsw->cur_element_count;
    }

    return comm().all_reduce_sum(hnsw_size);
  }

  const hnswlib::HierarchicalNSW<dist_t> &get_cell_hnsw(const int cell) const {
    return *m_voronoi_cell_hnsw[local_cell_index(cell)];
  }

  const index_vec_t &get_cell_pointers(index_t index) {
    return m_map_point_to_cells[index];
  }

  const point_t &get_point(index_t index) {
    ASSERT_RELEASE(m_local_data.count(index) > 0);
    return m_local_data[index];
  }

 private:
  hnswlib::HierarchicalNSW<dist_t> &get_cell_hnsw_non_const(
      const int cell) const {
    return *m_voronoi_cell_hnsw[local_cell_index(cell)];
  }

  void try_add_closest_cells(const index_t index, const index_vec_t &cells) {
    for (const index_t &cell : cells) {
      try_add_closest_cell(index, cell);
    }
  }

  // Scans through full list of cells to try inserting.
  // Can do something better if necessary, but this list should be small.
  void try_add_closest_cell(const index_t index, const index_t cell) {
    for (const index_t &current_cell : get_cell_pointers(index)) {
      if (current_cell == cell) return;
    }
    m_map_point_to_cells[index].push_back(cell);
  }

  void set_cell_add_vec_ordering(
      std::vector<std::pair<index_t, point_t>> &vec) {
#ifdef SALTATLAS_DETERMINISM  // Use fixed ordering if deterministic
    std::sort(vec.begin(), vec.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
#else  // Otherwise shuffle ordering
    std::random_shuffle(vec.begin(), vec.end());
#endif
  }

  data_index_cell_map_type m_map_point_to_cells;

  std::map<index_t, point_t> m_local_data;

  std::vector<hnswlib::HierarchicalNSW<dist_t> *> m_voronoi_cell_hnsw;
  hnswlib::SpaceInterface<dist_t>                *m_metric_space_ptr;

  std::vector<std::vector<std::pair<index_t, point_t>>>
      m_cell_add_vec;  // per-cell vector of indices to add to HNSW structure

  ygm::comm               *m_comm;
  ygm::ygm_ptr<dhnsw_impl> pthis;
  int                      m_comm_size;
  int                      m_comm_rank;

  partitioner_t &m_partitioner;

  int m_max_voronoi_rank;
  int m_num_cells;
};

}  // namespace dhnsw_detail
}  // namespace saltatlas
