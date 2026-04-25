// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/unordered/unordered_node_map.hpp>

#include <saltatlas/common/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/dnnd/detail/utilities/time.hpp>
#include <saltatlas/dnnd/utility.hpp>

#include <saltatlas/shm_knng_query/csr_knng.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/shm_knng_query/dense_point_store.hpp>
#include <saltatlas/shm_knng_query/query.hpp>

// External ID type and internal ID types.
#if defined(SALTATLAS_USE_STRING_ID)
using eid_type = saltatlas::pm_str_id_type;
using iid_type = uint32_t;
#else
using eid_type = uint32_t;
using iid_type = eid_type;
#endif

#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using dist_type =
    std::conditional_t<std::is_same_v<fe_type, double>, double, float>;

using point_store_type = saltatlas::dense_point_store<iid_type, fe_type>;

using knng_type = saltatlas::csr_knng<iid_type, std::size_t>;

using nn_query_kernel =
    saltatlas::knn_parallel_query_kernel<point_store_type, knng_type, iid_type,
                                         dist_type, fe_type>;

using e2i_id_map_type =
    boost::unordered::unordered_node_map<eid_type, uint32_t>;
using i2e_id_map_type =
    boost::unordered::unordered_node_map<iid_type, eid_type>;

struct option {
  nn_query_kernel::option query_option;
  std::filesystem::path   point_files_path;
  std::string             point_file_format;
  std::filesystem::path   index_files_path;
  std::string             distance_function;
  std::filesystem::path   query_file_path;
  std::vector<dist_type>  epsilons{0.0, 0.1, 0.2};
  std::filesystem::path   ground_truth_file_path;
  std::filesystem::path   query_result_file_path;

  /// Show the option values
  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "index_files_path: " << index_files_path << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "query_file_path: " << query_file_path << std::endl;
    std::cout << "ground_truth_file_path: " << ground_truth_file_path
              << std::endl;
    std::cout << "query_result_file_path: " << query_result_file_path
              << std::endl;
    std::cout << "n: " << query_option.k << std::endl;
    std::cout << "epsilons: ";
    for (const auto e : epsilons) {
      std::cout << e << " ";
    }
    std::cout << std::endl;
    std::cout << "query_option.verbose: " << query_option.verbose << std::endl;
  }
};

// parse CLI arguments
bool parse_options(int argc, char* argv[], option& opt, bool& show_help) {
  int c;
  while ((c = getopt(argc, argv, "i:g:p:f:q:n:e:G:o:M:vh")) != -1) {
    switch (c) {
      case 'i':
        opt.point_files_path = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'g':
        opt.index_files_path = optarg;
        break;

      case 'f':
        opt.distance_function = optarg;
        break;

      case 'q':
        opt.query_file_path = optarg;
        break;

      case 'n':
        opt.query_option.k = std::stoi(optarg);
        break;

      case 'e': {
        opt.epsilons = saltatlas::detail::str_split<dist_type>(optarg, ',');
        break;
      }
      case 'G':
        opt.ground_truth_file_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'v':
        opt.query_option.verbose = true;
        break;

      case 'h':
        show_help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        return false;
    }
  }

  if (opt.point_files_path.empty()) {
    std::cerr << "Point file path is not given." << std::endl;
    return false;
  }
  if (opt.point_file_format.empty()) {
    std::cerr << "Point file format is not given." << std::endl;
    return false;
  }
  if (opt.index_files_path.empty()) {
    std::cerr << "k-NN index file path is not given." << std::endl;
    return false;
  }
  if (opt.distance_function.empty()) {
    std::cerr << "Distance function name is not given." << std::endl;
    return false;
  }
  if (opt.query_file_path.empty()) {
    std::cerr << "Query file path is not given." << std::endl;
    return false;
  }
  if (opt.query_option.k == 0) {
    std::cerr << "k (number of nearest neighbors) must be > 0." << std::endl;
    return false;
  }
  if (opt.epsilons.empty()) {
    std::cerr << "No epsilon values are given." << std::endl;
    return false;
  }
  for (const auto e : opt.epsilons) {
    if (e < 0.0) {
      std::cerr << "Epsilon must be >= 0.0." << std::endl;
      return false;
    }
  }

  return true;
}

void show_usage(char* argv[]) {
  std::cout
      << "Usage: " << argv[0] << "\n[Required arguments]\n"
      << "-i string : Path to point file or directory that contain point "
         "files.\n"
      << "-p string : Point file format.\n"
      << "-g string : Path to k-NN index (graph) file or directory that "
         "contain index files.\n"
      << "-f string : Distance function name.\n"
      << "-q string : Path to a query file.\n"
      << "-n int    : Number of nearest neighbors to search for each query.\n"
      << "[Optional arguments]\n"
      << "-e string : Comma-separated epsilon values (default: "
         "0.0,0.1,0.2).\n"
      << "-G string : Path to a ground truth file.\n"
      << "-o string : Path to output query result file.\n"
      << "-v        : Verbose output.\n"
      << "-h       : Show this help message.\n"
      << std::endl;
}

e2i_id_map_type load_e2i_id_map(const std::filesystem::path& point_files_path) {
  e2i_id_map_type e2i_id_map;
  const auto      point_file_paths =
      saltatlas::dndetail::find_file_paths(point_files_path);
  for (const auto& file_path : point_file_paths) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open " << file_path << std::endl;
      std::abort();
    }

    for (std::string line; std::getline(ifs, line);) {
      std::stringstream ss(line);

      // External ID
      std::string buf;
      ss >> buf;
      eid_type eid = saltatlas::detail::str_cast<eid_type>(buf);

      // Internal ID
      iid_type iid;
      ss >> iid;

      const auto [itr, inserted] = e2i_id_map.emplace(eid, iid);
      if (!inserted) {
        std::cerr << "Duplicate external ID found in ID map: " << eid
                  << std::endl;
        std::abort();
      }
    }
  }
  return e2i_id_map;
}

e2i_id_map_type build_e2i_id_map(const std::filesystem::path& point_files_path,
                                 const std::string& point_file_format) {
  if (point_file_format != "wsv-id" && point_file_format != "tsv-id") {
    std::cerr << "Unsupported format: " << point_file_format << std::endl;
    std::abort();
  }

  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(point_files_path);
  const auto line_counts =
      saltatlas::smqdetail::count_lines_in_files(point_file_paths);

  std::vector<std::size_t> id_offsets(point_file_paths.size(), 0);
  for (std::size_t i = 1; i < point_file_paths.size(); ++i) {
    id_offsets[i] = id_offsets[i - 1] + line_counts[i - 1];
  }

  e2i_id_map_type global_e2i_id_map;
  global_e2i_id_map.reserve(
      std::accumulate(line_counts.begin(), line_counts.end(), std::size_t{0}));

  OMP_DIRECTIVE(parallel) {
    e2i_id_map_type local_e2i_id_map;

    for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
      const auto&   file_path = point_file_paths[i];
      std::ifstream ifs(file_path);
      if (!ifs.is_open()) {
        std::cerr << "Failed to open " << file_path << std::endl;
        std::abort();
      }

      std::size_t line_no = 0;
      for (std::string line; std::getline(ifs, line); ++line_no) {
        const auto iid = id_offsets[i] + line_no;

        eid_type          eid;
        std::string       token;
        std::stringstream ss(line);
        ss >> token;
        if (token.empty()) {
          std::cerr << "Failed to parse external ID from line: " << line
                    << std::endl;
          std::abort();
        }
        eid = saltatlas::detail::str_cast<eid_type>(token);

        const auto [itr, inserted] =
            local_e2i_id_map.emplace(std::move(eid), iid);
        if (!inserted) {
          std::cerr << "Duplicate external ID found while building ID map: "
                    << line << std::endl;
          std::abort();
        }
      }
    }

#pragma omp critical
    {
      global_e2i_id_map.insert(local_e2i_id_map.begin(),
                               local_e2i_id_map.end());
    }
  }

  return global_e2i_id_map;
}

i2e_id_map_type make_i2e_id_map(const e2i_id_map_type& e2i_id_map) {
  i2e_id_map_type i2e_id_map;
  for (const auto& [eid, iid] : e2i_id_map) {
    const auto [itr, inserted] = i2e_id_map.emplace(iid, eid);
    if (!inserted) {
      std::cerr << "Duplicate internal ID found in ID map: " << iid
                << std::endl;
      std::abort();
    }
  }
  return i2e_id_map;
}

int main(int argc, char* argv[]) {
  option opt;
  bool   show_help = false;
  if (!parse_options(argc, argv, opt, show_help)) {
    show_usage(argv);
    return EXIT_FAILURE;
  }
  if (show_help) {
    show_usage(argv);
    return EXIT_SUCCESS;
  }
  opt.show();
  std::cout << std::endl;

  std::optional<e2i_id_map_type> e2i_id_map = std::nullopt;
  std::optional<i2e_id_map_type> i2e_id_map = std::nullopt;
  if constexpr (!std::is_same_v<eid_type, iid_type>) {
    std::cout << "\nBuild external-to-internal ID map" << std::endl;
    // Construct external-to-internal ID map from point files if the map file
    // path is not given.
    e2i_id_map = build_e2i_id_map(opt.point_files_path, opt.point_file_format);
    i2e_id_map = make_i2e_id_map(*e2i_id_map);
  }

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  const auto points =
      saltatlas::load_points<eid_type, iid_type, fe_type, e2i_id_map_type>(
          point_file_paths, opt.point_file_format, e2i_id_map);
  std::cout << "Number of points: " << points.num_points() << std::endl;
  std::cout << "Number of dimensions: " << points.num_dimensions() << std::endl;

  std::cout << "\nLoad k-nn index" << std::endl;
  const auto index_file_paths =
      saltatlas::dndetail::find_file_paths(opt.index_files_path);
  const auto knng = [&]() {
    if constexpr (std::is_same_v<eid_type, iid_type>) {
      return knng_type(index_file_paths);
    } else {
      return knng_type(index_file_paths, *e2i_id_map);
    }
  }();
  std::cout << "Number of points: " << knng.num_points() << std::endl;
  std::cout << "Number of total neighbors: " << knng.num_total_neighbors()
            << std::endl;

  std::cout << "\nRead queries" << std::endl;
  std::vector<std::vector<fe_type>> queries;
  saltatlas::read_query(opt.query_file_path, queries);
  std::cout << "Number of queries: " << queries.size() << std::endl;

  std::cout << "\nStart query" << std::endl;
  for (const auto epsilon : opt.epsilons) {
    opt.query_option.epsilon = epsilon;
    nn_query_kernel kernel(opt.query_option, points,
                           opt.distance_function.c_str(), knng);
    std::cout << "epsilon: " << epsilon << std::endl;

    std::vector<std::vector<nn_query_kernel::neighbor_type>> results;
    double                                                   query_sec = 0.0;
    {
      const auto start_time = saltatlas::dndetail::get_time();
      results               = kernel.query(queries);
      query_sec             = saltatlas::dndetail::elapsed_time_sec(start_time);
    }
    std::cout << "Total query time (s): " << query_sec << std::endl;
    std::cout << "Throughput (qps): " << queries.size() / query_sec
              << std::endl;
    std::cout << "Mean query latency (ms): "
              << query_sec * 1000.0 / queries.size() << std::endl;

    if (!opt.query_result_file_path.empty()) {
      std::string path_str =
          opt.query_result_file_path.string() + ".e" + std::to_string(epsilon);

      if constexpr (std::is_same_v<eid_type, iid_type>) {
        std::cout << "Dump query results to " << path_str << std::endl;
        saltatlas::utility::dump_neighbors(results, path_str);
      } else {
        std::cout << "Dump query results with external IDs to " << path_str
                  << std::endl;
        std::vector<
            std::vector<saltatlas::detail::neighbor<eid_type, dist_type>>>
            results_eid(results.size());
        for (std::size_t i = 0; i < results.size(); ++i) {
          for (const auto& neighbor : results[i]) {
            const auto itr = i2e_id_map->find(neighbor.id);
            if (itr == i2e_id_map->end()) {
              std::cerr << "Internal ID " << neighbor.id
                        << " not found in internal-to-external ID map."
                        << std::endl;
              std::abort();
            }
            results_eid[i].emplace_back(itr->second, neighbor.distance);
          }
        }
        saltatlas::utility::dump_neighbors(results_eid, path_str);
      }
    }

    if (!opt.ground_truth_file_path.empty()) {
      std::vector<std::vector<nn_query_kernel::neighbor_type>> ground_truth;
      saltatlas::read_neighbors(opt.ground_truth_file_path, ground_truth);

      std::cout << "\nRecall scores" << std::endl;
      {
        const auto scores = saltatlas::utility::get_recall_scores(
            results, ground_truth, opt.query_option.k);
        std::cout << "Exact recall scores (min mean max): "
                  << *std::min_element(scores.begin(), scores.end()) << "\t"
                  << std::accumulate(scores.begin(), scores.end(), 0.0) /
                         scores.size()
                  << "\t" << *std::max_element(scores.begin(), scores.end())
                  << std::endl;
      }
      {
        const auto scores =
            saltatlas::utility::get_recall_scores_with_distance_ties(
                results, ground_truth, opt.query_option.k);
        std::cout << "Distance-tied recall scores (min mean max): "
                  << *std::min_element(scores.begin(), scores.end()) << "\t"
                  << std::accumulate(scores.begin(), scores.end(), 0.0) /
                         scores.size()
                  << "\t" << *std::max_element(scores.begin(), scores.end())
                  << std::endl;
      }
    }
    std::cout << std::endl;
  }
  std::cout << "\nAll done." << std::endl;

  return 0;
}
