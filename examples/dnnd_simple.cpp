// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using the DNND's simple API.
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_simple_example

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// ----- Point Type ----- //
using point_type = saltatlas::feature_vector<float>;

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  // Create a DNND object
  // Use the squared L2 distance function
  saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::distance::id::sql2,
                                              comm);

  // Load points from file(s)
  // The file format is assumed to be whitespace-separated values (wsv)
  // One point per line. Each feature value is separated by a whitespace.
  // DNND assigns an ID to each point in the order they are loaded,
  // i.e., ID is the line number starting from 0.
  std::vector<std::filesystem::path> paths{
      "../examples/datasets/point_5-4.txt"};
  g.load_points(paths.begin(), paths.end(), "wsv");

  // ----- NNG build and NN search APIs ----- //
  int k = 4;
  g.build(k);

  bool make_graph_undirected = true;
  g.optimize(make_graph_undirected);

  // Run queries
  {
    std::vector<point_type> queries;
    if (comm.rank() == 0) {
      queries.push_back(point_type{61.58, 29.68, 20.43, 99.22, 21.81});
    } else if (comm.rank() == 1) {
      queries.push_back(point_type{78.44, 54.43, 59.68, 65.80, 24.361});
    }

    {
      int        num_to_search = 4;
      const auto results =
          g.query(queries.begin(), queries.end(), num_to_search);
      // Show the query results
      comm.cout0() << "Query owner rank: neighbours (id, distance)..."
                   << std::endl;
      for (int i = 0; i <= 1; ++i) {
        if (comm.rank() == i) {
          std::cout << "Rank " << i << ": ";
          for (const auto& [nn_id, nn_dist] : results[0]) {
            std::cout << " " << nn_id << " (" << nn_dist << ")";
          }
          std::cout << std::endl;
        }
        comm.cf_barrier();
      }
    }

    // Get nearest neighbours with features
    comm.cout0()<< "\nNearest neighbor query result with features\n(show only the nearest point of the query from rank 0)" << std::endl;
    {
      int        num_to_search = 4;
      const auto results =
          g.query_with_features(queries.begin(), queries.end(), num_to_search);
      // Show the query results
      if (comm.rank() == 0) {
        auto& neighbours = results.first;
        auto& features   = results.second;
        std::cout << "Point ID " << neighbours[0][0].id << ", distance "
                     << neighbours[0][0].distance << ", feature {";
        for (const auto& v : features[0][0]) {
          std::cout << v << " ";
        }
        std::cout << "}" << std::endl;
      }
    }
  }

  // --- Point Data Accessors --- //
  comm.cout0() << "\nPoint 0's features: " << std::endl;
  {
    id_t pid = 0;
    if (g.contains_local(pid)) {
      auto p0 = g.get_local_point(pid);
      for (const auto& v : p0) {
        std::cout << v << " ";
      }
      std::cout << std::endl;
    }
  }

  comm.cout0() << "\nRank 0's all local points" << std::endl;
  {
    for (const auto& [id, point] : g.local_points()) {
      comm.cout0() << "Point ID " << id << " : ";
      for (const auto& v : point) {
        comm.cout0() << v << " ";
      }
      comm.cout0() << std::endl;
    }
  }

  comm.cout0() << "\nGet points including the ones that are stored in other ranks"
               << std::endl;
  {
    id_t ids[]  = {0, 1};
    auto points = g.get_points(ids, ids + 2);
    for (const auto& [id, point] : points) {
      comm.cout0() << "Point ID " << id << " : ";
      for (const auto& v : point) {
        comm.cout0() << v << " ";
      }
      comm.cout0() << std::endl;
    }
  }

  // Dump a KNNG to files
  g.dump_graph("./knng");
  comm.cout0() << "KNNG dumped to ./knng" << std::endl;

  return 0;
}
