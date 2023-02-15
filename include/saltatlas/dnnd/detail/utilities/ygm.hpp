// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iterator>
#include <type_traits>

#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <ygm/comm.hpp>

namespace saltatlas::dndetail {
/// \brief Distributes elements in a container.
/// This functions assumes that container whose iterator is incrementable and
/// has push_back() function.
/// Splits the input container by block,
/// i.e., the i-th rank gets the elements in the i-th block from the top.
/// \tparam container_type Container type.
/// \param source_container Source container.
/// \param target_container Container to stored distributed elements.
/// \param comm YGM comm.
template <typename container_type>
inline void distribute_elements_by_block(const container_type &source_container,
                                         container_type       &target_container,
                                         ygm::comm            &comm) {
  ygm::ygm_ptr<container_type> ptr_container(&target_container);
  comm.cf_barrier();

  auto itr = std::begin(source_container);
  for (int r = 0; r < comm.size(); ++r) {
    const auto range =
        dndetail::partial_range(source_container.size(), r, comm.size());
    container_type send_buf;
    for (std::size_t i = range.first; i < range.second; ++i) {
      send_buf.push_back(*itr++);
    }
    comm.async(
        r,
        [](auto ptr_container, const auto &recv_buf) {
          for (const auto &element : recv_buf)
            ptr_container->push_back(element);
        },
        ptr_container, send_buf);
  }
  comm.barrier();
}
}  // namespace saltatlas::dndetail