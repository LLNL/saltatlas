// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A small DNND advanced example that combines Metall persistence with
/// string IDs.
/// Usage:
///     cd build
///     mpirun -n 2 ./examples/dnnd_advanced_str_id

#include <mpi.h>

#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

using pm_id_type = saltatlas::pm_id_type;

namespace {

using point_type = saltatlas::pm_feature_vector<float>;
using index_type =
    saltatlas::dnnd_adv<pm_id_type, point_type, double, saltatlas::str_hash<>>;
using dataset_type        = std::vector<std::pair<pm_id_type, point_type>>;
using neighbor_store_type = typename index_type::neighbor_store_type;

// Keep all example inputs together so each phase reads as a short scenario.
struct example_data {
  dataset_type            base_dataset;
  dataset_type            extra_dataset;
  dataset_type            full_dataset;
  std::vector<point_type> base_queries;
  std::vector<point_type> full_queries;
};

constexpr std::uint64_t k_seed         = 20260416;
constexpr int           k_graph_degree = 3;
constexpr int           k_query_degree = 2;

[[noreturn]] void fail(ygm::comm& comm, const std::string& message) {
  if (comm.rank0()) {
    std::cerr << "ERROR: " << message << std::endl;
  }
  MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  std::abort();
}

void require(ygm::comm& comm, const bool condition,
             const std::string& message) {
  if (!condition) {
    fail(comm, message);
  }
}

pm_id_type make_id(const std::string_view text) {
  return pm_id_type(text.data(), text.size());
}

std::string to_std_string(const pm_id_type& id) {
  return std::string(id.data(), id.size());
}

point_type make_point(const std::initializer_list<float> values) {
  point_type point;
  point.reserve(values.size());
  point.insert(point.end(), values.begin(), values.end());
  return point;
}

std::string point_to_string(const point_type& point) {
  std::ostringstream oss;
  oss << '[';
  for (std::size_t i = 0; i < point.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << point[i];
  }
  oss << ']';
  return oss.str();
}

dataset_type make_base_dataset() {
  return {{make_id("alpha"), make_point({0.0F, 0.0F})},
          {make_id("bravo"), make_point({0.2F, 0.0F})},
          {make_id("charlie"), make_point({10.0F, 0.0F})},
          {make_id("delta"), make_point({10.2F, 0.0F})},
          {make_id("echo"), make_point({20.0F, 0.0F})},
          {make_id("foxtrot"), make_point({20.2F, 0.0F})}};
}

dataset_type make_extra_dataset() {
  return {{make_id("golf"), make_point({30.0F, 0.0F})},
          {make_id("hotel"), make_point({30.2F, 0.0F})}};
}

dataset_type append_dataset(dataset_type lhs, const dataset_type& rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

std::vector<point_type> make_queries(const bool include_extra_cluster) {
  std::vector<point_type> queries;
  queries.push_back(make_point({0.1F, 0.0F}));
  queries.push_back(make_point({10.1F, 0.0F}));
  queries.push_back(make_point({20.1F, 0.0F}));
  if (include_extra_cluster) {
    queries.push_back(make_point({30.1F, 0.0F}));
  }
  return queries;
}

example_data make_example_data() {
  example_data data;
  data.base_dataset  = make_base_dataset();
  data.extra_dataset = make_extra_dataset();
  data.full_dataset  = append_dataset(data.base_dataset, data.extra_dataset);
  data.base_queries  = make_queries(false);
  data.full_queries  = make_queries(true);
  return data;
}

std::vector<pm_id_type> collect_ids(const dataset_type& dataset) {
  std::vector<pm_id_type> ids;
  ids.reserve(dataset.size());
  for (const auto& [id, point] : dataset) {
    (void)point;
    ids.push_back(id);
  }
  return ids;
}

dataset_type make_local_slice(ygm::comm& comm, const dataset_type& dataset) {
  dataset_type local_dataset;
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    // Distribute example input explicitly so every rank contributes points.
    if (static_cast<int>(i % comm.size()) == comm.rank()) {
      local_dataset.push_back(dataset[i]);
    }
  }
  return local_dataset;
}

void add_dataset(ygm::comm& comm, index_type& index,
                 const dataset_type& dataset) {
  // dnnd_adv expects every rank to participate in collective operations.
  const auto local_dataset = make_local_slice(comm, dataset);
  const auto ids           = collect_ids(local_dataset);

  std::vector<point_type> points;
  points.reserve(local_dataset.size());
  for (const auto& [id, point] : local_dataset) {
    (void)id;
    points.push_back(point);
  }

  index.add_points(ids.begin(), ids.end(), points.begin(), points.end());
}

void remove_store(ygm::comm& comm, const std::filesystem::path& path) {
  if (comm.rank0()) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  comm.barrier();
}

void print_section(ygm::comm& comm, const std::string_view title) {
  comm.cout0() << "\n== " << title << " ==\n";
}

void print_points(ygm::comm& comm, index_type& index, const dataset_type& data,
                  const std::string_view label) {
  const auto ids    = collect_ids(data);
  const auto points = index.get_points(ids.begin(), ids.end());

  // num_points() is collective, so fetch it before rank-0-only printing.
  const size_t n = index.num_points();

  if (comm.rank0()) {
    std::cout << "\n" << label << "\n";
    std::cout << "total points: " << n << '\n';
    for (const auto& id : ids) {
      std::cout << "  " << to_std_string(id) << " -> "
                << point_to_string(points.at(id)) << '\n';
    }
  }
  comm.cf_barrier();
}

void print_query_results(ygm::comm&                     comm,
                         const std::vector<point_type>& queries,
                         const neighbor_store_type&     results,
                         const std::string_view         label) {
  if (comm.rank0()) {
    std::cout << "\n" << label << "\n";
    for (std::size_t i = 0; i < queries.size(); ++i) {
      std::cout << "query " << i << " " << point_to_string(queries[i])
                << " -> ";
      for (std::size_t j = 0; j < results[i].size(); ++j) {
        if (j != 0) {
          std::cout << ", ";
        }
        std::cout << to_std_string(results[i][j].id) << " ("
                  << results[i][j].distance << ")";
      }
      std::cout << '\n';
    }
  }
  comm.cf_barrier();
}

void print_query_results_with_features(
    ygm::comm& comm, const std::vector<point_type>& queries,
    const std::pair<neighbor_store_type, std::vector<std::vector<point_type>>>&
                           results,
    const std::string_view label) {
  if (!comm.rank0()) {
    return;
  }

  std::cout << "\n" << label << "\n";
  for (std::size_t i = 0; i < queries.size(); ++i) {
    std::cout << "query " << i << " " << point_to_string(queries[i]) << '\n';
    for (std::size_t j = 0; j < results.first[i].size(); ++j) {
      std::cout << "  " << to_std_string(results.first[i][j].id) << " ("
                << results.first[i][j].distance
                << ") feature=" << point_to_string(results.second[i][j])
                << '\n';
    }
  }
}

std::size_t create_base_index(ygm::comm& comm, index_type& index,
                              const example_data& data) {
  // Build one index over the original six points.
  add_dataset(comm, index, data.base_dataset);

  const auto index_id =
      index.build(saltatlas::distance::id::sql2, k_graph_degree);

  print_points(comm, index, data.base_dataset, "stored points");

  print_query_results(comm, data.base_queries,
                      index.query(index_id, saltatlas::distance::id::sql2,
                                  data.base_queries.begin(),
                                  data.base_queries.end(), k_query_degree),
                      "query results");
  return index_id;
}

void run_create_only_phase(ygm::comm&                   comm,
                           const std::filesystem::path& datastore_path,
                           const std::filesystem::path& snapshot_path,
                           const example_data&          data) {
  print_section(comm, "Phase 1: create_only");

  // Create a fresh Metall datastore and populate it.
  index_type index(saltatlas::create_only, datastore_path, comm, k_seed);

  create_base_index(comm, index, data);

  // The snapshot captures the six-point state before the later update step.
  require(comm, index.snapshot(snapshot_path), "snapshot() failed");
  if (comm.rank0()) {
    std::cout << "snapshot: " << snapshot_path << '\n';
  }
}

void run_open_only_phase(ygm::comm&                   comm,
                         const std::filesystem::path& datastore_path,
                         const example_data&          data) {
  print_section(comm, "Phase 2: open_only");

  // Reopen the same datastore, add more points, then refresh index 0.
  index_type index(saltatlas::open_only, datastore_path, comm, k_seed);
  add_dataset(comm, index, data.extra_dataset);
  index.update(0, saltatlas::distance::id::sql2, k_graph_degree);

  print_points(comm, index, data.full_dataset,
               "points after adding another cluster");
  print_query_results(
      comm, data.full_queries,
      index.query(0, saltatlas::distance::id::sql2, data.full_queries.begin(),
                  data.full_queries.end(), k_query_degree),
      "updated query results");
}

void run_open_read_only_phase(ygm::comm&                   comm,
                              const std::filesystem::path& snapshot_path,
                              const example_data&          data) {
  print_section(comm, "Phase 3: open_read_only");

  // Open the snapshot read-only to show that it still sees the earlier state.
  index_type index(saltatlas::open_read_only, snapshot_path, comm, k_seed);
  print_query_results_with_features(
      comm, data.base_queries,
      index.query_with_features(0, saltatlas::distance::id::sql2,
                                data.base_queries.begin(),
                                data.base_queries.end(), k_query_degree),
      "query results from the snapshot");
}

}  // namespace

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  const std::filesystem::path datastore_path{"/tmp/saltatlas-dnnd-str-id"};
  const std::filesystem::path snapshot_path{
      "/tmp/saltatlas-dnnd-str-id-snapshot"};

  const auto data = make_example_data();

  // Start from a clean slate so repeated runs behave the same way.
  remove_store(comm, datastore_path);
  remove_store(comm, snapshot_path);

  if (comm.rank0()) {
    std::cout << "Metall datastore: " << datastore_path << std::endl;
  }
  comm.cf_barrier();

  run_create_only_phase(comm, datastore_path, snapshot_path, data);
  run_open_only_phase(comm, datastore_path, data);
  run_open_read_only_phase(comm, snapshot_path, data);

  remove_store(comm, datastore_path);
  remove_store(comm, snapshot_path);
  return 0;
}
