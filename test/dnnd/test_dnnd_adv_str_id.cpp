#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/string.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

using pm_id_type = saltatlas::pm_id_type;

namespace {

using point_type = saltatlas::pm_feature_vector<double>;
using index_type =
    saltatlas::dnnd_adv<pm_id_type, point_type, double, saltatlas::str_hash<>>;
using neighbor_type       = typename index_type::neighbor_type;
using neighbor_store_type = typename index_type::neighbor_store_type;
using dataset_type        = std::vector<std::pair<pm_id_type, point_type>>;
using point_table_type =
    std::unordered_map<pm_id_type, point_type, saltatlas::str_hash<>>;
using initial_index_map_type = typename index_type::external_initial_index_type;

constexpr std::uint64_t k_seed         = 20260416;
constexpr int           k_graph_degree = 3;
constexpr int           k_query_degree = 2;

std::string to_std_string(const pm_id_type &id) {
  return std::string(id.data(), id.size());
}

pm_id_type make_id(const std::string_view text) {
  return pm_id_type(text.data(), text.size());
}

point_type make_point(const std::initializer_list<double> values) {
  point_type point;
  point.reserve(values.size());
  point.insert(point.end(), values.begin(), values.end());
  return point;
}

dataset_type make_base_dataset() {
  return {{make_id("alpha"), make_point({0.0, 0.0})},
          {make_id("bravo"), make_point({0.2, 0.0})},
          {make_id("charlie"), make_point({10.0, 0.0})},
          {make_id("delta"), make_point({10.2, 0.0})},
          {make_id("echo"), make_point({20.0, 0.0})},
          {make_id("foxtrot"), make_point({20.2, 0.0})}};
}

dataset_type make_extra_dataset() {
  return {{make_id("golf"), make_point({30.0, 0.0})},
          {make_id("hotel"), make_point({30.2, 0.0})}};
}

dataset_type append_dataset(dataset_type lhs, const dataset_type &rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

dataset_type make_local_dataset_slice(ygm::comm          &comm,
                                      const dataset_type &dataset) {
  dataset_type local_dataset;
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    if (static_cast<int>(i % comm.size()) == comm.rank()) {
      local_dataset.push_back(dataset[i]);
    }
  }
  return local_dataset;
}

std::vector<point_type> collect_points(const dataset_type &dataset) {
  std::vector<point_type> points;
  points.reserve(dataset.size());
  for (const auto &[id, point] : dataset) {
    (void)id;
    points.push_back(point);
  }
  return points;
}

std::vector<point_type> make_queries(const bool include_extra) {
  std::vector<point_type> queries;
  queries.emplace_back(make_point({0.1, 0.0}));
  queries.emplace_back(make_point({10.1, 0.0}));
  queries.emplace_back(make_point({20.1, 0.0}));
  if (include_extra) {
    queries.emplace_back(make_point({30.1, 0.0}));
  }
  return queries;
}

std::vector<std::vector<pm_id_type>> make_expected_query_groups(
    const bool include_extra) {
  std::vector<std::vector<pm_id_type>> groups;
  groups.push_back({make_id("alpha"), make_id("bravo")});
  groups.push_back({make_id("charlie"), make_id("delta")});
  groups.push_back({make_id("echo"), make_id("foxtrot")});
  if (include_extra) {
    groups.push_back({make_id("golf"), make_id("hotel")});
  }
  return groups;
}

std::vector<pm_id_type> collect_ids(const dataset_type &dataset) {
  std::vector<pm_id_type> ids;
  ids.reserve(dataset.size());
  for (const auto &[id, point] : dataset) {
    (void)point;
    ids.push_back(id);
  }
  return ids;
}

void add_dataset(ygm::comm &comm, index_type &index,
                 const dataset_type &dataset) {
  const auto local_dataset = make_local_dataset_slice(comm, dataset);
  const auto ids           = collect_ids(local_dataset);
  const auto points        = collect_points(local_dataset);
  index.add_points(ids.begin(), ids.end(), points.begin(), points.end());
}

point_table_type make_point_table(const dataset_type &dataset) {
  point_table_type table;
  table.reserve(dataset.size());
  for (const auto &[id, point] : dataset) {
    table.emplace(id, point);
  }
  return table;
}

initial_index_map_type make_external_initial_index(
    const dataset_type &dataset) {
  const auto             ids = collect_ids(dataset);
  initial_index_map_type initial_index;
  initial_index.reserve(ids.size());
  initial_index.emplace(ids[0],
                        std::vector<pm_id_type>{ids[1], ids[2], ids[3]});
  initial_index.emplace(ids[1],
                        std::vector<pm_id_type>{ids[0], ids[2], ids[3]});
  initial_index.emplace(ids[2],
                        std::vector<pm_id_type>{ids[3], ids[0], ids[1]});
  initial_index.emplace(ids[3],
                        std::vector<pm_id_type>{ids[2], ids[0], ids[1]});
  initial_index.emplace(ids[4],
                        std::vector<pm_id_type>{ids[5], ids[2], ids[3]});
  initial_index.emplace(ids[5],
                        std::vector<pm_id_type>{ids[4], ids[2], ids[3]});
  return initial_index;
}

std::string join_ids(const std::vector<neighbor_type> &neighbors) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << to_std_string(neighbors[i].id);
  }
  return oss.str();
}

void print_progress(ygm::comm &comm, const std::string &message) {
  if (comm.rank0()) {
    std::printf("[progress] %s\n", message.c_str());
    std::fflush(stdout);
  }
}

[[noreturn]] void fail(ygm::comm &comm, const std::string &message) {
  if (comm.rank0()) {
    std::cerr << "FAIL: " << message << std::endl;
  }
  MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
  std::abort();
}

void require(ygm::comm &comm, const bool condition,
             const std::string &message) {
  if (!condition) {
    fail(comm, message);
  }
}

void require_point_eq(ygm::comm &comm, const point_type &lhs,
                      const point_type &rhs, const std::string &message) {
  require(comm, lhs == rhs, message);
}

void require_ids_eq(ygm::comm &comm, const std::vector<std::size_t> &lhs,
                    const std::vector<std::size_t> &rhs,
                    const std::string              &message) {
  require(comm, lhs == rhs, message);
}

bool contains_id(const std::vector<neighbor_type> &neighbors,
                 const std::vector<pm_id_type>    &candidates) {
  for (const auto &neighbor : neighbors) {
    for (const auto &candidate : candidates) {
      if (neighbor.id == candidate) {
        return true;
      }
    }
  }
  return false;
}

std::vector<neighbor_type> sorted_neighbors(
    std::vector<neighbor_type> neighbors) {
  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

void require_feature_alignment(ygm::comm                        &comm,
                               const std::vector<neighbor_type> &neighbors,
                               const std::vector<point_type>    &features,
                               const point_table_type           &point_table,
                               const std::string &message_prefix) {
  require(comm, neighbors.size() == features.size(),
          message_prefix + ": neighbor/feature count mismatch");
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    const auto it = point_table.find(neighbors[i].id);
    require(comm, it != point_table.end(),
            message_prefix + ": missing point for neighbor " +
                to_std_string(neighbors[i].id));
    require_point_eq(comm, features[i], it->second,
                     message_prefix + ": feature mismatch for neighbor " +
                         to_std_string(neighbors[i].id));
  }
}

void check_query_results(
    ygm::comm &comm, const neighbor_store_type &results,
    const point_table_type                     &point_table,
    const std::vector<std::vector<pm_id_type>> &expected_groups,
    const std::string                          &label) {
  require(comm, results.size() == expected_groups.size(),
          label + ": unexpected number of query results");
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (comm.rank0()) {
      std::printf("[debug] %s query %zu neighbors: %s\n", label.c_str(), i,
                  join_ids(results[i]).c_str());
    }
    require(comm, results[i].size() == k_query_degree,
            label + ": unexpected k for query " + std::to_string(i));
    for (const auto &neighbor : results[i]) {
      require(comm, point_table.contains(neighbor.id),
              label + ": unknown neighbor " + to_std_string(neighbor.id));
    }
    require(comm, contains_id(results[i], expected_groups[i]),
            label + ": expected cluster not found in result " +
                std::to_string(i) + " (got " + join_ids(results[i]) + ")");
  }
}

void check_query_with_features(
    ygm::comm &comm, const neighbor_store_type &results,
    const std::vector<std::vector<point_type>> &features,
    const point_table_type                     &point_table,
    const std::vector<std::vector<pm_id_type>> &expected_groups,
    const std::string                          &label) {
  check_query_results(comm, results, point_table, expected_groups, label);
  require(comm, results.size() == features.size(),
          label + ": result/feature outer size mismatch");
  for (std::size_t i = 0; i < results.size(); ++i) {
    require_feature_alignment(comm, results[i], features[i], point_table,
                              label + ": query " + std::to_string(i));
  }
}

void check_dataset_state(ygm::comm &comm, const index_type &index,
                         const dataset_type &dataset,
                         const std::string  &label) {
  const auto point_table = make_point_table(dataset);
  const auto ids         = collect_ids(dataset);

  require(comm, index.num_points() == dataset.size(),
          label + ": unexpected global point count");

  const auto fetched_points = index.get_points(ids.begin(), ids.end());
  require(comm, fetched_points.size() == dataset.size(),
          label + ": get_points returned an unexpected number of entries");
  for (const auto &[id, point] : dataset) {
    const auto it = fetched_points.find(id);
    require(comm, it != fetched_points.end(),
            label + ": missing point " + to_std_string(id));
    require_point_eq(comm, it->second, point,
                     label + ": get_points mismatch for " + to_std_string(id));

    const auto local_count = ygm::sum(index.contains_local(id) ? 1 : 0, comm);
    require(comm, local_count == 1,
            label + ": point ownership mismatch for " + to_std_string(id));
    if (index.contains_local(id)) {
      require_point_eq(
          comm, index.get_local_point(id), point,
          label + ": get_local_point mismatch for " + to_std_string(id));
    }
  }

  std::size_t local_range_count = 0;
  for (const auto &[internal_id, point] : index.local_points()) {
    (void)internal_id;
    (void)point;
    ++local_range_count;
  }

  std::size_t local_iterator_count = 0;
  for (auto it = index.local_points_begin(); it != index.local_points_end();
       ++it) {
    ++local_iterator_count;
  }

  require(comm, local_range_count == index.num_local_points(),
          label + ": local_points range count mismatch");
  require(comm, local_iterator_count == index.num_local_points(),
          label + ": local_points iterator count mismatch");
  require(comm, ygm::sum(local_range_count, comm) == index.num_points(),
          label + ": local point counts do not sum to global count");
}

void check_neighbors_api(ygm::comm &comm, const index_type &index,
                         const std::size_t   index_id,
                         const dataset_type &dataset,
                         const std::string  &label) {
  const auto point_table = make_point_table(dataset);
  const auto ids         = collect_ids(dataset);

  std::cerr << "[debug] " << __LINE__ << " " << label
            << ": checking neighbors API with index ID " << index_id
            << std::endl;

  const auto neighbors_table =
      index.get_neighbors(index_id, ids.begin(), ids.end());
  require(comm, neighbors_table.size() == dataset.size(),
          label + ": get_neighbors returned an unexpected number of rows");

  std::cerr << "[debug] " << __LINE__ << " " << label
            << ": get_neighbors returned " << neighbors_table.size()
            << " rows (expected " << dataset.size() << ")" << std::endl;

  const auto neighbors_with_features =
      index.get_neighbors_with_features(index_id, ids.begin(), ids.end());
  require(comm, neighbors_with_features.size() == dataset.size(),
          label +
              ": get_neighbors_with_features returned an unexpected number of "
              "rows");

  std::cerr << "[debug] " << __LINE__ << " " << label
            << ": get_neighbors_with_features returned "
            << neighbors_with_features.size() << " rows (expected "
            << dataset.size() << ")" << std::endl;

  for (const auto &[id, point] : dataset) {
    (void)point;
    const auto neighbors_it = neighbors_table.find(id);
    require(comm, neighbors_it != neighbors_table.end(),
            label + ": missing neighbors for " + to_std_string(id));
    require(comm, !neighbors_it->second.empty(),
            label + ": empty neighbor list for " + to_std_string(id));

    const auto feature_it = neighbors_with_features.find(id);
    require(comm, feature_it != neighbors_with_features.end(),
            label + ": missing neighbor features for " + to_std_string(id));

    require(comm,
            sorted_neighbors(neighbors_it->second) ==
                sorted_neighbors(feature_it->second.first),
            label +
                ": get_neighbors and get_neighbors_with_features differ for " +
                to_std_string(id));
    require_feature_alignment(
        comm, feature_it->second.first, feature_it->second.second, point_table,
        label + ": neighbor features for " + to_std_string(id));

    if (index.contains_local(id)) {
      require(
          comm,
          neighbors_it->second.size() ==
              index.num_local_neighbors(index_id, id),
          label + ": num_local_neighbors mismatch for " + to_std_string(id));
    }
  }
}

std::vector<std::filesystem::path> write_wsv_dataset(
    ygm::comm &comm, const dataset_type &dataset,
    const std::filesystem::path &dir, const std::string &stem) {
  if (comm.rank0()) {
    std::filesystem::create_directories(dir);
    const auto                       midpoint = dataset.size() / 2;
    const std::array<std::size_t, 3> offsets{0, midpoint, dataset.size()};
    for (std::size_t part = 0; part < 2; ++part) {
      std::ofstream out(dir / (stem + "_" + std::to_string(part) + ".txt"));
      for (std::size_t i = offsets[part]; i < offsets[part + 1]; ++i) {
        out << to_std_string(dataset[i].first);
        for (const auto value : dataset[i].second) {
          out << ' ' << value;
        }
        out << '\n';
      }
    }
  }
  comm.barrier();

  return {dir / (stem + "_0.txt"), dir / (stem + "_1.txt")};
}

std::vector<std::filesystem::path> write_custom_dataset(
    ygm::comm &comm, const dataset_type &dataset,
    const std::filesystem::path &dir, const std::string &stem) {
  if (comm.rank0()) {
    std::filesystem::create_directories(dir);
    const auto                       midpoint = dataset.size() / 2;
    const std::array<std::size_t, 3> offsets{0, midpoint, dataset.size()};
    for (std::size_t part = 0; part < 2; ++part) {
      std::ofstream out(dir / (stem + "_" + std::to_string(part) + ".txt"));
      for (std::size_t i = offsets[part]; i < offsets[part + 1]; ++i) {
        out << to_std_string(dataset[i].first) << '|' << dataset[i].second[0]
            << '|' << dataset[i].second[1] << '\n';
      }
    }
  }
  comm.barrier();

  return {dir / (stem + "_0.txt"), dir / (stem + "_1.txt")};
}

std::pair<pm_id_type, point_type> parse_custom_line(const std::string &line) {
  std::stringstream ss{line};
  std::string       token;
  std::getline(ss, token, '|');
  const auto id = make_id(token);
  std::getline(ss, token, '|');
  const auto x = std::stod(token);
  std::getline(ss, token, '|');
  const auto y = std::stod(token);
  return {id, make_point({x, y})};
}

void run_load_points_format_suite(ygm::comm                   &comm,
                                  const std::filesystem::path &root,
                                  const bool use_metall_runtime) {
  const std::string label = use_metall_runtime
                                ? "persistent load_points(format)"
                                : "non-persistent load_points(format)";
  print_progress(comm, "starting " + label);
  if (comm.rank0()) {
    std::filesystem::create_directories(root);
  }
  comm.barrier();

  const auto dataset = make_base_dataset();
  const auto files =
      write_wsv_dataset(comm, dataset, root / "inputs", "wsv_id");
  print_progress(comm, label + ": dataset files written");

  if (use_metall_runtime) {
    index_type index(saltatlas::create_only, root / "metall_load_format", comm,
                     k_seed, false);
    index.load_points(files.begin(), files.end(), "wsv-id");
    check_dataset_state(comm, index, dataset, label);
    const auto index_id =
        index.build(saltatlas::distance::id::sql2, k_graph_degree);
    print_progress(comm, label + ": index built");
    check_neighbors_api(comm, index, index_id, dataset, label);
  } else {
    index_type index(comm, k_seed, false);
    index.load_points(files.begin(), files.end(), "wsv-id");
    check_dataset_state(comm, index, dataset, label);
    const auto index_id =
        index.build(saltatlas::distance::id::sql2, k_graph_degree);
    print_progress(comm, label + ": index built");
    check_neighbors_api(comm, index, index_id, dataset, label);
  }
  print_progress(comm, "finished " + label);
}

void run_load_points_parser_suite(ygm::comm                   &comm,
                                  const std::filesystem::path &root,
                                  const bool use_metall_runtime) {
  const std::string label = use_metall_runtime
                                ? "persistent load_points(parser)"
                                : "non-persistent load_points(parser)";
  print_progress(comm, "starting " + label);
  if (comm.rank0()) {
    std::filesystem::create_directories(root);
  }
  comm.barrier();

  const auto dataset = make_base_dataset();
  const auto files =
      write_custom_dataset(comm, dataset, root / "inputs", "custom_parser");
  print_progress(comm, label + ": dataset files written");
  const std::function<std::pair<pm_id_type, point_type>(const std::string &)>
      parser = [](const std::string &line) { return parse_custom_line(line); };

  if (use_metall_runtime) {
    index_type index(saltatlas::create_only, root / "metall_load_parser", comm,
                     k_seed, false);
    index.load_points(files.begin(), files.end(), parser);
    check_dataset_state(comm, index, dataset, label);
    const auto index_id =
        index.build(saltatlas::distance::id::sql2, k_graph_degree);
    print_progress(comm, label + ": index built");
    check_neighbors_api(comm, index, index_id, dataset, label);
  } else {
    index_type index(comm, k_seed, false);
    index.load_points(files.begin(), files.end(), parser);
    check_dataset_state(comm, index, dataset, label);
    const auto index_id =
        index.build(saltatlas::distance::id::sql2, k_graph_degree);
    print_progress(comm, label + ": index built");
    check_neighbors_api(comm, index, index_id, dataset, label);
  }
  print_progress(comm, "finished " + label);
}

struct suite_state {
  dataset_type             full_dataset;
  std::vector<std::size_t> index_ids;
};

suite_state run_core_suite(ygm::comm &comm, index_type &index,
                           const std::filesystem::path &root,
                           const std::string           &label,
                           const bool expect_snapshot_failure) {
  print_progress(comm, "starting " + label);
  if (comm.rank0()) {
    std::filesystem::create_directories(root);
  }
  comm.barrier();

  const auto distance_func =
      saltatlas::distance::distance_function<point_type, double>(
          saltatlas::distance::id::sql2);

  const auto base_dataset  = make_base_dataset();
  const auto extra_dataset = make_extra_dataset();
  const auto full_dataset  = append_dataset(base_dataset, extra_dataset);

  add_dataset(comm, index, base_dataset);
  check_dataset_state(comm, index, base_dataset, label + ": base dataset");
  print_progress(comm, label + ": base dataset loaded");

  const auto initial_index = make_external_initial_index(base_dataset);

  const auto index_id_dist =
      index.build(saltatlas::distance::id::sql2, k_graph_degree);
  const auto index_id_func = index.build(distance_func, k_graph_degree);
  const auto index_id_external_dist =
      index.build(saltatlas::distance::id::sql2, k_graph_degree, initial_index);
  const auto index_id_external_func =
      index.build(distance_func, k_graph_degree, initial_index);
  print_progress(comm, label + ": built initial indices");

  require_ids_eq(comm, index.get_index_ids(), {0, 1, 2, 3},
                 label + ": unexpected index IDs after builds");

  const auto base_queries = make_queries(false);
  const auto base_groups  = make_expected_query_groups(false);
  const auto base_points  = make_point_table(base_dataset);

  const auto query_dist =
      index.query(index_id_dist, saltatlas::distance::id::sql2,
                  base_queries.begin(), base_queries.end(), k_query_degree);
  check_query_results(comm, query_dist, base_points, base_groups,
                      label + ": query(distance id)");

  const auto query_dist_features = index.query_with_features(
      index_id_dist, saltatlas::distance::id::sql2, base_queries.begin(),
      base_queries.end(), k_query_degree);
  check_query_with_features(
      comm, query_dist_features.first, query_dist_features.second, base_points,
      base_groups, label + ": query_with_features(distance id)");

  const auto query_func =
      index.query(index_id_func, distance_func, base_queries.begin(),
                  base_queries.end(), k_query_degree);
  check_query_results(comm, query_func, base_points, base_groups,
                      label + ": query(function)");

  const auto query_func_features = index.query_with_features(
      index_id_func, distance_func, base_queries.begin(), base_queries.end(),
      k_query_degree);
  check_query_with_features(
      comm, query_func_features.first, query_func_features.second, base_points,
      base_groups, label + ": query_with_features(function)");

  const std::array<std::size_t, 2> multi_index_ids{index_id_dist,
                                                   index_id_func};

  const auto multi_query_dist =
      index.query(multi_index_ids.begin(), multi_index_ids.end(),
                  saltatlas::distance::id::sql2, base_queries.begin(),
                  base_queries.end(), k_query_degree);
  check_query_results(comm, multi_query_dist, base_points, base_groups,
                      label + ": multi query(distance id)");

  const auto multi_query_dist_features = index.query_with_features(
      multi_index_ids.begin(), multi_index_ids.end(),
      saltatlas::distance::id::sql2, base_queries.begin(), base_queries.end(),
      k_query_degree);
  check_query_with_features(comm, multi_query_dist_features.first,
                            multi_query_dist_features.second, base_points,
                            base_groups,
                            label + ": multi query_with_features(distance id)");

  const auto multi_query_func =
      index.query(multi_index_ids.begin(), multi_index_ids.end(), distance_func,
                  base_queries.begin(), base_queries.end(), k_query_degree);
  check_query_results(comm, multi_query_func, base_points, base_groups,
                      label + ": multi query(function)");

  const auto multi_query_func_features = index.query_with_features(
      multi_index_ids.begin(), multi_index_ids.end(), distance_func,
      base_queries.begin(), base_queries.end(), k_query_degree);
  check_query_with_features(comm, multi_query_func_features.first,
                            multi_query_func_features.second, base_points,
                            base_groups,
                            label + ": multi query_with_features(function)");
  print_progress(comm, label + ": base queries verified");

  check_neighbors_api(comm, index, index_id_dist, base_dataset,
                      label + ": neighbors(distance id)");
  check_neighbors_api(comm, index, index_id_external_dist, base_dataset,
                      label + ": neighbors(external initial index)");

  index.optimize(index_id_external_dist, saltatlas::distance::id::sql2);
  index.optimize(index_id_external_func, distance_func);
  print_progress(comm, label + ": optimize completed");

  add_dataset(comm, index, extra_dataset);
  check_dataset_state(comm, index, full_dataset, label + ": full dataset");
  print_progress(comm, label + ": extra dataset loaded");

  index.update(index_id_dist, saltatlas::distance::id::sql2, k_graph_degree);
  index.update(index_id_func, distance_func, k_graph_degree);

  const auto full_queries = make_queries(true);
  const auto full_groups  = make_expected_query_groups(true);
  const auto full_points  = make_point_table(full_dataset);
  const auto updated_query =
      index.query(index_id_dist, saltatlas::distance::id::sql2,
                  full_queries.begin(), full_queries.end(), k_query_degree);
  check_query_results(comm, updated_query, full_points, full_groups,
                      label + ": updated query(distance id)");

  check_neighbors_api(comm, index, index_id_dist, full_dataset,
                      label + ": updated neighbors(distance id)");
  print_progress(comm, label + ": updated queries verified");

  index.erase(index_id_external_func);
  require_ids_eq(comm, index.get_index_ids(), {0, 1, 2},
                 label + ": unexpected index IDs after erase");
  print_progress(comm, label + ": erased external index");

  if (expect_snapshot_failure) {
    require(comm, !index.snapshot(root / "unexpected_snapshot"),
            label + ": snapshot() should fail without Metall runtime");
    print_progress(comm, label + ": snapshot failure path verified");
  }

  print_progress(comm, "finished " + label);
  return {full_dataset, index.get_index_ids()};
}

void check_reopened_suite(ygm::comm &comm, index_type &index,
                          const suite_state &state, const std::string &label) {
  print_progress(comm, "starting " + label);
  check_dataset_state(comm, index, state.full_dataset, label + ": dataset");
  require_ids_eq(comm, index.get_index_ids(), state.index_ids,
                 label + ": unexpected index IDs");
  check_neighbors_api(comm, index, state.index_ids.front(), state.full_dataset,
                      label + ": neighbors");

  const auto point_table = make_point_table(state.full_dataset);
  const auto queries     = make_queries(true);
  const auto groups      = make_expected_query_groups(true);
  const auto query_results =
      index.query(state.index_ids.front(), saltatlas::distance::id::sql2,
                  queries.begin(), queries.end(), k_query_degree);
  check_query_results(comm, query_results, point_table, groups,
                      label + ": query(distance id)");

  const auto query_with_features = index.query_with_features(
      state.index_ids.front(), saltatlas::distance::id::sql2, queries.begin(),
      queries.end(), k_query_degree);
  check_query_with_features(comm, query_with_features.first,
                            query_with_features.second, point_table, groups,
                            label + ": query_with_features(distance id)");
  print_progress(comm, "finished " + label);
}

void remove_path_on_rank0(ygm::comm &comm, const std::filesystem::path &path) {
  if (comm.rank0()) {
    std::filesystem::remove_all(path);
  }
  comm.barrier();
}

}  // namespace

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  print_progress(comm, "starting test_dnnd_adv_str_id");

  const auto root =
      std::filesystem::current_path() / "test_dnnd_adv_str_id_artifacts";
  remove_path_on_rank0(comm, root);
  if (comm.rank0()) {
    std::filesystem::create_directories(root);
  }
  comm.barrier();
  print_progress(comm, std::string("artifacts root: ") + root.string());

  run_load_points_format_suite(comm, root / "non_persistent_format", false);
  run_load_points_parser_suite(comm, root / "non_persistent_parser", false);

  {
    index_type index(comm, k_seed, false);
    run_core_suite(comm, index, root / "non_persistent_core",
                   "non-persistent core", true);
  }

  run_load_points_format_suite(comm, root / "persistent_format", true);
  run_load_points_parser_suite(comm, root / "persistent_parser", true);

  const auto persistent_store = root / "persistent_core_store";
  const auto snapshot_store   = root / "persistent_snapshot_store";
  const auto copy_store       = root / "persistent_copy_store";
  remove_path_on_rank0(comm, persistent_store);
  remove_path_on_rank0(comm, snapshot_store);
  remove_path_on_rank0(comm, copy_store);

  suite_state persistent_state;
  {
    index_type index(saltatlas::create_only, persistent_store, comm, k_seed,
                     false);
    persistent_state = run_core_suite(comm, index, root / "persistent_core",
                                      "persistent core", false);
    print_progress(comm, "persistent core: creating snapshot");
    require(comm, index.snapshot(snapshot_store),
            "persistent core: snapshot() failed");
  }

  print_progress(comm, "persistent core: copying closed store");
  require(comm, index_type::copy(persistent_store, copy_store, comm),
          "persistent core: copy() failed");

  {
    index_type index(saltatlas::open_only, snapshot_store, comm, k_seed, false);
    check_reopened_suite(comm, index, persistent_state,
                         "persistent open_only snapshot");
  }

  {
    index_type index(saltatlas::open_read_only, copy_store, comm, k_seed,
                     false);
    check_reopened_suite(comm, index, persistent_state,
                         "persistent open_read_only copy");
  }

  comm.barrier();
  remove_path_on_rank0(comm, root);
  print_progress(comm, "finished test_dnnd_adv_str_id");
  if (comm.rank0()) {
    std::cout << "PASS" << std::endl;
  }
  return 0;
}
