// Copyright 2020-2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include <metall/detail/time.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>

// Parse CLI arguments
// s: source data store path
// d: destination data store path
bool parse_option(int argc, char **argv, std::string &s, std::string &d) {
  int c;
  while ((c = getopt(argc, argv, "s:d:")) != -1) {
    switch (c) {
      case 's':
        s = optarg;
        break;
      case 'd':
        d = optarg;
        break;
      default:
        return false;
    }
  }
  return !s.empty() && !d.empty();
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  std::string from;
  std::string to;
  if (!parse_option(argc, argv, from, to)) {
    comm.cerr0()
        << "Usage: " << argv[0]
        << " -s <source datastore path> -d <destination datastore path>"
        << std::endl;
    ::MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  }

  comm.cout0() << "Transfer PM (Metall) datastore from " << from << " to " << to
               << std::endl;
  const auto start = metall::mtlldetail::elapsed_time_sec();
  const auto ret   = metall::utility::metall_mpi_adaptor::copy(
      from.c_str(), to.c_str(), comm.get_mpi_comm());
  const auto elapsed_time_sec = metall::mtlldetail::elapsed_time_sec(start);

  if (ret) {
    comm.cout0() << "Finished transfer (s): " << elapsed_time_sec << std::endl;
  } else {
    comm.cerr0() << "Failed to transfer." << std::endl;
    ::MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}