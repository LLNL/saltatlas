// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <assert.h>
#include <cstdint>
#include <iostream>

namespace saltatlas {
namespace dhnsw_detail {

class logical_physical_mapper {
 public:
  using physical_id_type = int;
  using logical_id_type  = int;
  using local_id_type    = int16_t;

  logical_physical_mapper(const logical_id_type  num_logical_partitions,
                          const physical_id_type num_physical_partitions)
      : m_num_logical_partitions(num_logical_partitions),
        m_num_physical_partitions(num_physical_partitions) {}

  /*
   * @brief Determine the physical partition (i.e. MPI rank) responsible for
   * storing a logical partition
   *
   * @param logical The logical ID of a partition
   *
   * @return The physical partition ID where the logical partition can be found
   */
  physical_id_type logical_to_physical(const logical_id_type logical) const {
    assert(logical < m_num_logical_partitions);
    return logical % m_num_physical_partitions;
  }

  /*
   * @brief Determine the local ID of a partition within its physical partition
   * (i.e. MPI rank)
   *
   * @param logical The logical ID of a partition
   *
   * @return The local ID of the partition
   */
  local_id_type logical_to_local(const logical_id_type logical) const {
    assert(logical < m_num_logical_partitions);
    return logical / m_num_physical_partitions;
  }

  /*
   * @brief Determine the number of local partitions held on a physical
   * partition
   *
   * @param physical The physical ID to determine the local size of
   *
   * @return The number of logical partitions stored on the given physical
   * partition
   */
  local_id_type local_size(const physical_id_type physical) const {
    if (physical >= m_num_physical_partitions) {
      std::cout << physical << "\t" << m_num_physical_partitions << std::endl;
    }
    assert(physical < m_num_physical_partitions);
    return m_num_logical_partitions / m_num_physical_partitions +
           (physical < (m_num_logical_partitions % m_num_physical_partitions));
  }

 private:
  int m_num_logical_partitions;
  int m_num_physical_partitions;
};
}  // namespace dhnsw_detail
}  // namespace saltatlas
