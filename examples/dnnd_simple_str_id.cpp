// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of using the DNND's simple API.
/// Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_simple

#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd.hpp>

// Point ID type
using id_type   = std::string;
using dist_type = double;

// ----- Point Type ----- //
using point_type        = saltatlas::feature_vector<float>;
using string_point_type = saltatlas::feature_vector<char>;

std::string gen_point_id(int rank, int i) {
  return "point_" + std::to_string(rank) + "-" + std::to_string(i);
}

point_type gen_point(int rank, int i) {
  return point_type{static_cast<float>(100.0 * rank + 10 * i + 0.1),
                    static_cast<float>(100.0 * rank + 10 * i + 0.2)};
}

point_type gen_query_point(int rank, int i) {
  return point_type{static_cast<float>(100.0 * rank + 10 * i + 1.1),
                    static_cast<float>(100.0 * rank + 10 * i + 1.2)};
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  {
    saltatlas::dnnd<id_type, point_type, dist_type, saltatlas::str_hash<>> g(
        saltatlas::distance::id::sql2, comm);

    std::vector<std::string> ids;
    std::vector<point_type>  points;
    for (int i = 0; i < 4 * (comm.rank() + 1); ++i) {
      ids.push_back(gen_point_id(comm.rank(), i));
      points.push_back(gen_point(comm.rank(), i));
    }
    g.add_points(ids.begin(), ids.end(), points.begin(), points.end());

    comm.cout0() << "#of total points" << " = " << g.num_points() << std::endl;

    // Check the location of a point
    const auto id_0_0 = gen_point_id(0, 0);
    if (g.contains_local(id_0_0)) {
      std::cout << id_0_0 << " is in rank " << comm.rank() << std::endl;
      auto p0 = g.get_local_point(id_0_0);
      assert(p0 == gen_point(0, 0));
    }

    // test stored points
    {
      // Generate point IDs to check.
      std::vector<id_type> test_ids;
      for (int r = 0; r < comm.size(); ++r) {
        for (int i = 0; i < 4 * (r + 1); ++i) {
          test_ids.push_back(gen_point_id(r, i));
        }
      }

      // Check if all ranks can access all points and get the correct point
      // data.
      auto points = g.get_points(test_ids.begin(), test_ids.end());
      for (int r = 0; r < comm.size(); ++r) {
        for (int i = 0; i < 4 * (r + 1); ++i) {
          const auto point = gen_point(r, i);
          assert(point == gen_point(r, i));
        }
      }
    }

    // Construct a KNNG
    int k = 2;
    g.build(k);
    g.optimize(true, 1.5);

    // Dump the index to a file
    g.dump_index("./dnnd_simple_str_id_index");

    // Get KNNG neighbors of a point
    {
      const auto pid = gen_point_id(1 % comm.size(), 3);
      if (g.contains_local(pid)) {
        auto neighbors = g.get_local_neighbors(pid);
        std::cout << "Neighbors of " << pid << " in rank " << comm.rank()
                  << ": ";
        for (const auto& neighbor : neighbors) {
          std::cout << "{" << neighbor << "}, ";
        }
        std::cout << std::endl;
      }
      comm.cf_barrier();

      // Get neighbors including the ones that could be stored in remote ranks
      comm.cout0() << "\nGet neighbors of " << pid << " from all ranks"
                   << std::endl;
      const auto neighbors = g.get_neighbors(ids.begin(), ids.end());
      for (const auto& [id, neighbors] : neighbors) {
        comm.cout0() << "Neighbors of " << id << ": ";
        for (const auto& neighbor : neighbors) {
          comm.cout0() << "{" << neighbor << "}, ";
        }
        comm.cout0() << std::endl;
      }

      // Get neighbors with features
      comm.cout0() << "\nGet neighbors with features of " << pid
                   << " from all ranks" << std::endl;
      const auto neighbors_with_features =
          g.get_neighbors_with_features(ids.begin(), ids.end());
      for (const auto& [id, neighbors_and_features] : neighbors_with_features) {
        const auto& neighbors = neighbors_and_features.first;
        const auto& features  = neighbors_and_features.second;
        comm.cout0() << "Neighbors of " << id << ": ";
        for (size_t i = 0; i < neighbors.size(); ++i) {
          comm.cout0() << "{" << neighbors[i]
                       << ", feature: " << saltatlas::to_string(features[i])
                       << "}, ";
        }
        comm.cout0() << std::endl;
      }
    }  // get-neighbors examples

    // Query
    {
      std::vector<point_type> queries{gen_query_point(comm.rank(), 0),
                                      gen_query_point(comm.rank(), 1)};
      const auto query_result = g.query(queries.begin(), queries.end(), 4);
      comm.cout0() << "\nQuery result for " << queries.size()
                   << " query points: " << std::endl;
      for (size_t i = 0; i < query_result.size(); ++i) {
        comm.cout0() << "Query: " << saltatlas::to_string(queries[i])
                     << ", Neighbors: ";
        for (const auto& neighbor : query_result[i]) {
          comm.cout0() << "{" << neighbor << "}, ";
        }
        comm.cout0() << std::endl;
      }
    }

    // Query (results with features)
    {
      // Generate queries.
      // NOTE: all ranks must submit the same queries
      std::vector<point_type> queries{gen_query_point(0, 0),
                                      gen_query_point(0, 1)};

      const auto query_result_with_features =
          g.query_with_features(queries.begin(), queries.end(), 4);
      const auto& query_result      = query_result_with_features.first;
      const auto& neighbor_features = query_result_with_features.second;
      comm.cout0() << "\nQuery with features result for " << queries.size()
                   << " query points: " << std::endl;
      for (size_t i = 0; i < query_result.size(); ++i) {
        comm.cout0() << "Query: " << saltatlas::to_string(queries[i])
                     << ", Neighbors: ";
        for (size_t j = 0; j < query_result[i].size(); ++j) {
          comm.cout0() << "{" << query_result[i][j] << ", feature: "
                       << saltatlas::to_string(neighbor_features[i][j])
                       << "}, ";
        }
        comm.cout0() << std::endl;
      }
    }  // query with features examples
  }
  comm.barrier();

  {
    comm.cout0() << "\nLoad points from a file and build an index" << std::endl;
    saltatlas::dnnd<id_type, point_type, dist_type, saltatlas::str_hash<>> g(
        saltatlas::distance::id::sql2, comm);
    std::vector<std::filesystem::path> files{
        ".//examples/datasets/point_5-4_str-id.txt"};
    g.load_points(files.begin(), files.end(), "wsv-id");
    g.build(2);
    g.dump_index("./dnnd_simple_str_id_index_from_file");
  }

  {
    comm.cout0() << "\nLoad points from a file with a custom line parser"
                 << std::endl;
    saltatlas::dnnd<id_type, point_type, dist_type, saltatlas::str_hash<>> g(
        saltatlas::distance::id::sql2, comm);
    std::vector<std::filesystem::path> files{
        ".//examples/datasets/point_5-4_str-id.txt"};

    // Manually parse each line of the file.
    const auto line_parser = [](const std::string& line) {
      std::stringstream ss(line);
      id_type           id;
      ss >> id;
      point_type point(4);
      for (size_t i = 0; i < point.size(); ++i) {
        ss >> point[i];
      }
      return std::make_pair(id, point);
    };

    g.load_points(files.begin(), files.end(), line_parser);
    g.build(2);
    g.dump_index("./dnnd_simple_str_id_index_from_file_custom_parser");
  }

  {
    comm.cout0() << "\nLoad string points with string ID" << std::endl;
    saltatlas::dnnd<id_type, string_point_type, dist_type,
                    saltatlas::str_hash<>>
        g(saltatlas::distance::id::levenshtein, comm);
    std::vector<std::filesystem::path> files{
        ".//examples/datasets/point_string_str-id.txt"};
    g.load_points(files.begin(), files.end(), "str-id");
    g.build(3);
    g.optimize(true, 1.5);
    g.dump_index("./dnnd_simple_str_id_index_from_file_string_point");
  }

  return 0;
}
