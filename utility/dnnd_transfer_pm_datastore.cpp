// Copyright 2020-2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include <metall/detail/time.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>

int main(int argc, char **argv) {
  ::MPI_Init(&argc, &argv);
  {
    const int rank = metall::utility::mpi::comm_rank(MPI_COMM_WORLD);
    if (rank == -1) {
      std::cerr << "Failed to get MPI rank" << std::endl;
      ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (argc != 3 && rank == 0) {
      std::cerr << ("Usage: ", argv[0], " copy_from_path copy_to_path");
      ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const auto from = argv[1];
    const auto to   = argv[2];
    if (rank == 0) {
      std::cout << "Transfer PM (Metall) datastore from " << from << " to "
                << to << std::endl;
    }
    const auto start = metall::mtlldetail::elapsed_time_sec();
    const auto ret =
        metall::utility::metall_mpi_adaptor::copy(from, to, MPI_COMM_WORLD);
    const auto elapsed_time_sec = metall::mtlldetail::elapsed_time_sec(start);
    if (rank == 0) {
      if (ret) {
        std::cout << "Finished transfer (s): " << elapsed_time_sec << std::endl;
      } else {
        std::cerr << "Failed to transfer." << std::endl;
        ::MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }
  }
  ::MPI_Finalize();

  return EXIT_SUCCESS;
}