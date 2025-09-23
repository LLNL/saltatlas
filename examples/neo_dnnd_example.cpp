// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Must run srun with '-mblock' option, which is the default one.
// Do not use '--mpibind=off' option.

#define METALL_DISABLE_CONCURRENCY

#ifdef __APPLE__
#define METALL_DEFAULT_CAPACITY (1ULL << 30ULL)
#else
#define METALL_DEFAULT_CAPACITY (1ULL << 36ULL)
#endif

#include <filesystem>
#include <iostream>
#include <string>

#include "saltatlas/neo_dnnd/neo_dnnd.hpp"

using namespace saltatlas;

using id_type = uint32_t;
using feature_elem_type = float;
using distance_type = double;
using neo_dnnd_t = neo_dnnd<id_type, feature_elem_type, distance_type>;
using point_type = typename neo_dnnd_t::point_type;

int main(int argc, char* argv[]) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    mpi::communicator comm;

    auto l2_func = distance::distance_function<point_type, distance_type>("l2");
    neo_dnnd_t dnnd(l2_func, comm);
    comm.barrier();

    dnnd.read_dataset("./examples/datasets/point_5-4.txt", "wsv");

    const int k = 4;
    auto knng = dnnd.construct(k);

    comm.cout0() << "Optimize KNNG" << std::endl;
    const double pruning_factor = -1;  // No pruning
    dnnd.optimize(pruning_factor, knng);

    std::filesystem::path dump_path = "neo_dnnd_knng_dump";
    comm.cout0() << "\nDump to " << dump_path << std::endl;
    std::error_code ec;
    std::filesystem::create_directories(dump_path, ec);
    comm.barrier();
    const bool dump_distance = true;
    dnnd.dump_graph(knng, dump_path / "knng", dump_distance);

    comm.barrier();
  }
  ::MPI_Finalize();

  return 0;
}