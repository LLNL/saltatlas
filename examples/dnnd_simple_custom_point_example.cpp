// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// ----- Point Type ----- //
// using point_type = saltatlas::feature_vector<double>;
// Custom point type (just alias of STL-containers)
// using point_type = std::vector<std::vector<double>>;

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

  {
    std::vector<id_t>        ids;
    std::vector<graph_point> points;
    g.add_points(ids.begin(), ids.end(), points.begin(), points.end());
  }

  {
    std::vector<std::filesystem::path> paths{
        "../examples/datasets/point_5-4.txt"};
    g.load_points(paths.begin(), paths.end(), [](const std::string& line) {
      id_t        id;
      graph_point p;
      return std::make_pair(id, p);
    });
  }
  // ----- Point Data Accessors ----- //
  id_t id = 0;
  g.get_local_point(id);
  g.local_points_begin();
  g.local_points_end();

  // ----- NNG build and NN search APIs ----- //
  // Build KNNG, only for DNND?
  int    k     = 10;
  double rho   = 0.8;
  double delta = 0.001;
  g.build(k, rho, delta);

  bool make_graph_undirected = true;
  g.optimize(make_graph_undirected);

  // Run queries
  std::vector<graph_point> queries;
  int                      num_to_search = 10;
  double                   epsilon       = 0.1;
  const auto               results =
      g.query(queries.begin(), queries.end(), num_to_search, 0.1);

  // Dump a KNNG to files
  // If 'aggregate' is true, dump all data to a single file in rank 0.
  g.dump_graph("./knng");

  return 0;
}
