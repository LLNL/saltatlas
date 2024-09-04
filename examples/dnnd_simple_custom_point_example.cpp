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
#include <random>
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

// Randomly generate a point, just for demonstration
graph_point gen_point() {
  graph_point        p;
  std::random_device rd;
  std::mt19937       gen(rd());
  const int          num_vertices = gen() % 10 + 1;
  p.data.resize(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    const int degree = gen() % 10 + 1;
    p.data[i].reserve(degree);
    for (size_t j = 0; j < degree; ++j) {
      p.data[i][j] = std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
  }
  return p;
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  saltatlas::dnnd<id_t, graph_point, dist_t> g(distance_func, comm);

  // Add points
  {
    std::vector<id_t>        ids;
    std::vector<graph_point> points;
    // Assuming ids and points are stored in vectors
    for (size_t i = 0; i < 10; ++i) {
      ids.push_back(i + 10 * comm.rank());
      points.push_back(gen_point());
    }
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
