// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <saltatlas/dhnsw/detail/dist_index.hpp>
#include <saltatlas/dhnsw/detail/functional.hpp>
#include <saltatlas/dhnsw/detail/query_engine.hpp>

namespace saltatlas {

template <typename DistType, typename Point, typename Partitioner>
class dhnsw {
 public:
  dhnsw(int max_voronoi_rank, int num_cells,
        hnswlib::SpaceInterface<DistType> *space_ptr, ygm::comm *comm,
        Partitioner &p)
      : m_comm(comm),
        m_index_impl(max_voronoi_rank, num_cells, space_ptr, comm, p),
        m_query_engine_impl(&m_index_impl){};

  void partition_data(ygm::container::bag<std::pair<uint64_t, Point>> &data,
                      const uint32_t num_partitions) {
    m_index_impl.partition_data(data, num_partitions);
  }

  ~dhnsw() { m_comm->barrier(); }

  void queue_data_point_insertion(const size_t pt_idx, const Point &pt) {
    m_index_impl.add_data_point_to_insertion_queue(pt_idx, pt);
  }

  // This function is necessary because the hnswlib version used does not
  // support resizing HNSW structures. So the number of points to insert needs
  // to be known before building (unless you want to risk running out of
  // memory). I believe this limitation is gone in newer versions of hnswlib, so
  // adding points may be much simpler if this is updated.
  void initialize_hnsw() {
    m_comm->barrier();
    m_index_impl.initialize_hnsw();
    // m_index_impl.flush_insertion_queues();
  }

  void fill_seed_hnsw() {
    m_comm->barrier();
    m_index_impl.fill_seed_hnsw();
  }

  /*
void set_seeds(const std::vector<Point> &seed_features) {
m_index_impl.store_seeds(seed_features);
}
  */

  template <typename Callback, typename... Callback_Args>
  void query(const Point &query_pt, const int k, const int hops,
             const int initial_queries, const int voronoi_rank, Callback c,
             const Callback_Args &...args) {
    m_query_engine_impl.query(query_pt, k, hops, initial_queries, voronoi_rank,
                              c, args...);
  }

  template <typename Function>
  void for_all_data(Function fn) {
    m_index_impl.for_all_data(fn);
  }

  size_t global_size() { return m_index_impl.global_size(); }

  inline ygm::comm &comm() { return *m_comm; }

 private:
  ygm::comm                                             *m_comm;
  dhnsw_detail::dhnsw_impl<DistType, Point, Partitioner> m_index_impl;
  dhnsw_detail::query_engine_impl<DistType, Point, Partitioner>
      m_query_engine_impl;
};

}  // namespace saltatlas
