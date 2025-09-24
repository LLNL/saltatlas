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

int main(int argc, char* argv[]) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    mpi::communicator comm;

    // Build and optimize knng from scratch
    {
      auto l2_func =
          distance::distance_function<point_type, distance_type>("l2");
      neo_dnnd_t dnnd(l2_func, comm);

      comm.cout0() << "Load points from a file" << std::endl;
      dnnd.load_points("./examples/datasets/point_5-4.txt", "wsv");

      comm.cout0() << "Build a KNNG" << std::endl;
      const int k    = 4;
      auto      knng = dnnd.build(k);

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
    }
    comm.barrier();

    // Use add_points() function to add points
    comm.cout0() << "\nAdd points using add_points() function" << std::endl;
    {
      neo_dnnd_t dnnd(
          distance::distance_function<point_type, distance_type>("cosine"),
          comm);

      std::vector<id_type>                        ids;
      std::vector<std::vector<feature_elem_type>> points;
      if (comm.rank() == 0) {
        ids.push_back(0);
        ids.push_back(1);
        ids.push_back(2);
        points.push_back(
            std::vector<feature_elem_type>{1.0f, 0.0f, 0.0f, 0.0f});
        points.push_back(
            std::vector<feature_elem_type>{0.0f, 1.0f, 0.0f, 0.0f});
        points.push_back(
            std::vector<feature_elem_type>{0.0f, 0.0f, 1.0f, 0.0f});
      }
      if (comm.size() >= 2 && comm.rank() == 1) {
        ids.push_back(3);
        ids.push_back(4);
        points.push_back(
            std::vector<feature_elem_type>{0.0f, 0.0f, 0.0f, 1.0f});
        points.push_back(
            std::vector<feature_elem_type>{1.0f, 1.0f, 1.0f, 1.0f});
      }
      dnnd.add_points(ids.begin(), ids.end(), points.begin(), points.end());

      dnnd.build(2);
    }
    comm.barrier();
  }
  ::MPI_Finalize();

  return 0;
}