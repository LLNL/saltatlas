// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <vector>

namespace saltatlas {
namespace dhnsw_detail {

// This class holds the collections of pointers to nearby partitions from any of
// the data points our indices are built from
template <typename IndexType, typename PartitionIDType = IndexType>
class data_partition_recommender {
 public:
  using id_type           = IndexType;
  using partition_id_type = PartitionIDType;

  data_partition_recommender() {};

  void add_partition_

      private : std::unordered_map<id_type, std::vector<partition_id_type>>
                    m_partition_recommendations;
};
}  // namespace dhnsw_detail
}  // namespace saltatlas
