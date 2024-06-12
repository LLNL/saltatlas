// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using DNND's simple API with a custom point type
/// and custom distance function.
/// It is recommended to see the examples/dnnd_simple_example.cpp beforehand.
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_simple_custom_point_example

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// Custom point type
struct graph_point {
 public:
  std::vector<std::vector<double>> data;

  // For cereal
  template <class Archive>
  void serialize(Archive& ar) {
    ar(data);
  }
};

// Custom distance function
dist_t distance_func(const graph_point& a, const graph_point& b) {
  // Custom distance function
  return a.data.size() + b.data.size();
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  saltatlas::dnnd<id_t, graph_point, dist_t> g(distance_func, comm);

  // Add points
  {
    // Assuming ids and points are stored in vectors
    std::vector<id_t>        ids;
    std::vector<graph_point> points;
    g.add_points(ids.begin(), ids.end(), points.begin(), points.end());
  }

  // Load points using a custom data parser
  {
    std::vector<std::filesystem::path> paths{};
    g.load_points(paths.begin(), paths.end(), [](const std::string& line) {
      id_t        id;
      graph_point p;
      return std::make_pair(id, p);
    });
  }

  int k = 10;
  g.build(k);

  g.optimize();

  std::vector<graph_point> queries;
  int                      num_to_search = 10;
  const auto results = g.query(queries.begin(), queries.end(), num_to_search);

  return 0;
}
