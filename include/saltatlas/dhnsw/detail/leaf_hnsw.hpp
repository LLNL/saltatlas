// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>
#include <saltatlas/dhnsw/detail/metric_hyperplane_partitioner.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

namespace saltatlas {
namespace dhnsw_detail {

template <typename Distance, typename Point>
class leaf_hnsw {
 public:
  using point_type    = Point;
  using distance_type = Distance;
  using hnsw_type     = hnswlib::HierarchicalNSW<distance_type>;

  leaf_hnsw(
      dhnsw_detail::SpaceWrapper<distance_type, point_type> &space_wrapper,
      const int num_partitions, const size_t M, const size_t ef_construction,
      const size_t seed)
      : m_hnsw(&space_wrapper, num_partitions, M, ef_construction, seed) {}

  template <typename IndexType>
  void add_leaf_points(metric_hyperplane_partitioner<distance_type, IndexType,
                                                     point_type> &partitioner) {
    const auto tree_leaves = partitioner.get_leaves();

    for (const auto &[partition_num, rep_point] : tree_leaves) {
      m_hnsw.addPoint(&(rep_point.get()), partition_num);
    }
  }

  std::priority_queue<std::pair<distance_type, size_t>> search(
      const point_type &point, const int k) {
    return m_hnsw.searchKnn(&point, k);
  }

 private:
  hnsw_type m_hnsw;
};
}  // namespace dhnsw_detail
}  // namespace saltatlas
