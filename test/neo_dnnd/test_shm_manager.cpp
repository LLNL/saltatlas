// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include <iostream>
#include <saltatlas/common/detail/utilities/backtrace.hpp>
#include <saltatlas/neo_dnnd/detail/utilities/shm_manager.hpp>
#include <saltatlas/neo_dnnd/mpi.hpp>

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  signal(SIGSEGV, show_backtrace);
#endif

  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    saltatlas::mpi::communicator comm;
    auto shm_mems =
        saltatlas::dndetail::create_and_open_shm<int>("shm-test", 4096, comm);

    const auto lc_rank = comm.node_local_rank();
    auto* ptr = shm_mems[lc_rank].get();
    for (size_t i = 0; i < shm_mems[lc_rank].size(); ++i) {
      ptr[i] = static_cast<int>(comm.rank() * 1000 + i);
    }
    comm.node_local_barrier();

    // Local root checks all shared memory regions
    if (lc_rank == 0) {
      for (int r = 0; r < comm.node_size(); ++r) {
        auto* ptr = shm_mems[r].get();
        const auto owner = comm.rank() + r;
        for (size_t i = 0; i < shm_mems[r].size(); ++i) {
          const auto expected = static_cast<int>(owner * 1000 + i);
          if (ptr[i] != expected) {
            comm.cerr() << "Data mismatch at index " << i << ": expected "
                        << expected << ", got " << ptr[i] << std::endl;
            comm.abort();
          }
        }
      }
    }
    comm.barrier();
    comm.cout0() << "Success!" << std::endl;
  }
  ::MPI_Finalize();

  return 0;
}