// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <memory>
#include <vector>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename Point, typename PointID>
class local_hnsw {
 public:
  using distance_type = DistType;
  using point_type    = Point;
  using hnsw_type     = hnswlib::HierarchicalNSW<distance_type>;
  // using id_type       = PointID;
  using id_type =
      size_t;  // hnswlib hard-codes labeltype to size_t. Causes issues when
               // getting values from hnswlib if ours doesn't match here.

  // Initialize local HNSWs with space for 0 points
  local_hnsw(
      dhnsw_detail::SpaceWrapper<distance_type, point_type> &space_wrapper,
      const int num_local_partitions, const size_t M,
      const size_t ef_construction, const size_t seed) {
    for (int i = 0; i < num_local_partitions; ++i) {
      std::unique_ptr<hnsw_type> p_hnsw(
          new hnsw_type(&space_wrapper, 0, M, ef_construction, seed));
      m_hnsw_vec.push_back(std::move(p_hnsw));
    }
  }

  // Resize HNSW structures
  void resize(const std::vector<size_t> &hnsw_sizes) {
    assert(hnsw_sizes.size() == m_hnsw_vec.size());

    for (int i = 0; i < hnsw_sizes.size(); ++i) {
      m_hnsw_vec[i]->resizeIndex(hnsw_sizes[i]);
    }
  }

  void add_point(const point_type &point, const size_t point_id,
                 const size_t local_partition) {
    assert(local_partition < m_hnsw_vec.size());

    m_hnsw_vec[local_partition]->addPoint(&point, point_id);
  }

  std::priority_queue<std::pair<distance_type, id_type>> search(
      const point_type &point, const int k, const int local_partition) const {
    return m_hnsw_vec[local_partition]->searchKnn(&point, k);
  }

 private:
  // Need to store unique pointers to HNSWs because issues with constructors
  // don't let them be placed directly in a vector
  std::vector<std::unique_ptr<hnsw_type>> m_hnsw_vec;
};
}  // namespace dhnsw_detail
}  // namespace saltatlas
