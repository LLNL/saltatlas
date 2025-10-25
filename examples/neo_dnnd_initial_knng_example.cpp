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

using id_type           = uint32_t;
using feature_elem_type = float;
using distance_type     = double;
using neo_dnnd_t        = neo_dnnd<id_type, feature_elem_type, distance_type>;
using point_type        = typename neo_dnnd_t::point_type;
using knng_type         = typename neo_dnnd_t::knng_type;

int main(int argc, char* argv[]) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    mpi::communicator comm;

    knng_type initial_knng;
    {
      neo_dnnd_t dnnd(
          distance::distance_function<point_type, distance_type>("l2"), comm);

      std::vector<std::filesystem::path> paths{
          "./examples/datasets/point_5-4.txt"};
      dnnd.load_points(paths.begin(), paths.end(), "wsv");
      initial_knng = dnnd.build(2);
    }
    comm.barrier();

    {
      neo_dnnd_t dnnd(
          distance::distance_function<point_type, distance_type>("cosine"),
          comm, true);
      std::vector<std::filesystem::path> paths{
          "./examples/datasets/point_5-4.txt"};
      dnnd.load_points(paths.begin(), paths.end(), "wsv");
      auto knng = dnnd.build(4, 0.5, 0.001, 0.0, 1 << 25, initial_knng);

      std::error_code ec;
      std::filesystem::create_directories("second-knng", ec);
      comm.barrier();
      dnnd.dump_graph(knng, "second-knng/knng", true);
    }
    comm.barrier();
  }
  ::MPI_Finalize();

  return 0;
}