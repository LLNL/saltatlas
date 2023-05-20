// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iterator>
#include <type_traits>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/detail/utilities/general.hpp>
#include <saltatlas/dnnd/detail/utilities/mpi.hpp>

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

/// \briefe Send YGM's async messages in batch mode.
/// \tparam async_sender Message launcher function type.
/// Expected signature is: std::size_t(ygm::comm&).
/// \param num_local_items Number of local items.
/// \param global_batch_size Global batch size.
/// if 0 is specified, this function sends all items in one batch.
/// \param verbose Verbose flag.
/// \param sender Message sender function.
/// sender must return the number of sent items during the function call.
/// This function decide the timing of invoking YGM's barrier() based on the
/// returned values from the sender.
/// \param comm YGM comm.
template <typename async_sender>
inline void run_batched_ygm_async(const std::size_t   num_local_items,
                                  const std::size_t   global_batch_size,
                                  const bool          verbose,
                                  const async_sender &sender, ygm::comm &comm) {
  for (std::size_t num_sent = 0, batch_no = 0;; ++batch_no) {
    assert(num_local_items >= num_sent);
    const auto num_local_remains = num_local_items - num_sent;
    if (verbose) {
      comm.cout0() << "Batch #" << batch_no << std::endl;
      comm.cout0() << "#of remains: " << comm.all_reduce_sum(num_local_remains)
                   << std::endl;
    }

    const auto b = global_batch_size > 0
                       ? global_batch_size
                       : std::numeric_limits<std::size_t>::max();
    const auto local_batch_size =
        mpi::assign_tasks(num_local_remains, b, comm.rank(), comm.size(),
                          verbose, comm.get_mpi_comm());
    // Note: this algorithm does not check #of send items in a batch strictly,
    // letting the sender send more items than the batch size.
    for (std::size_t i = 0; i < local_batch_size;) {
      const auto n = std::invoke(sender, comm);
      i += n;
      num_sent += n;
    }
    assert(num_sent <= num_local_items &&
           "More items than initially told were sent.");
    comm.barrier();

    const auto finished =
        comm.all_reduce_min(num_sent == num_local_items ? 1 : 0);
    if (finished > 0) break;
  }
}

}  // namespace saltatlas::dndetail