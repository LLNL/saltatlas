// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage:
//   cd saltatlas/build
//   mpirun -n 2 ./examples/dnnd_example

#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <saltatlas/dnnd/dnnd_simple.hpp>
#include "dnnd_example_common.hpp"

#ifdef SALTATLAS_DNND_EXAMPLE_ID_TYPE
using id_type = SALTATLAS_DNND_EXAMPLE_ID_TYPE;
#else
using id_type = uint32_t;
#endif

#ifdef SALTATLAS_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE
using feature_element_type = SALTATLAS_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE;
#else
using feature_element_type = float;
#endif

#ifdef SALTATLAS_DNND_EXAMPLE_DISTANCE_TYPE
using distance_type = SALTATLAS_DNND_EXAMPLE_DISTANCE_TYPE;
#else
using distance_type =
    std::conditional_t<std::is_same_v<feature_element_type, double>, double,
                       float>;
#endif

using dnnd_type =
    saltatlas::dnnd<id_type, saltatlas::feature_vector<feature_element_type>,
                    distance_type>;

struct option_t {
  int                                index_k{0};
  double                             r{0.8};
  double                             delta{0.001};
  std::string                        distance_name;
  std::vector<std::filesystem::path> point_file_names;
  std::string                        point_file_format;
  std::size_t                        batch_size{1ULL << 25};
  bool                               make_index_undirected{false};
  double                             pruning_degree_multiplier{0.0};
  std::filesystem::path              query_file_path;
  std::filesystem::path              ground_truth_file_path;
  std::filesystem::path              query_result_file_path;
  int                                query_k{0};
  double                             epsilon{0.1};
  std::filesystem::path              index_dump_prefix;
  bool                               dump_index_with_distance{false};
  bool                               verbose{false};
};

bool parse_options(int, char **, option_t &, bool &);
template <typename cout_type>
void usage(std::string_view, cout_type &);
template <typename cout_type>
void show_options(const option_t &, cout_type &);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  show_config<id_type, feature_element_type, distance_type>(comm);

  option_t opt;
  bool     help{false};
  if (!parse_options(argc, argv, opt, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    return 0;
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }
  show_options(opt, comm.cout0());

  dnnd_type g(saltatlas::distance::convert_to_distance_id(opt.distance_name),
              comm, std::random_device{}(), opt.verbose);

  {
    comm.cout0() << "\n<<Read Points>>" << std::endl;
    const auto paths =
        saltatlas::utility::find_file_paths(opt.point_file_names);
    ygm::timer point_read_timer;
    g.load_points(paths.begin(), paths.end(), opt.point_file_format);
    comm.cout0() << "\nReading points took (s)\t" << point_read_timer.elapsed()
                 << std::endl;
    comm.cout0() << "#of points\t" << g.num_points() << std::endl;
  }

  {
    comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
    ygm::timer const_timer;
    g.build(opt.index_k, opt.r, opt.delta, opt.batch_size);
    comm.cout0() << "\nkNNG construction took (s)\t" << const_timer.elapsed()
                 << std::endl;
  }

  if (opt.make_index_undirected) {
    comm.cout0() << "\n<<kNNG Optimization>>" << std::endl;
    ygm::timer optimization_timer;
    g.optimize(opt.make_index_undirected, opt.pruning_degree_multiplier);
    comm.cout0() << "\nkNNG optimization took (s)\t"
                 << optimization_timer.elapsed() << std::endl;
  }

  if (!opt.query_file_path.empty()) {
    std::vector<dnnd_type::point_type> queries;
    saltatlas::read_query(opt.query_file_path, queries, comm);

    comm.cout0() << "Executing queries" << std::endl;
    ygm::timer step_timer;
    const auto query_results =
        g.query(queries.begin(), queries.end(), opt.query_k, opt.epsilon);
    comm.cf_barrier();
    comm.cout0() << "\nProcessing queries took (s)\t" << step_timer.elapsed()
                 << std::endl;

    if (!opt.ground_truth_file_path.empty()) {
      show_query_recall_score(query_results, opt.ground_truth_file_path, comm);
      show_query_recall_score_with_only_distance(
          query_results, opt.ground_truth_file_path, comm);
      show_query_recall_score_with_distance_ties(
          query_results, opt.ground_truth_file_path, comm);
    }

    if (!opt.query_result_file_path.empty()) {
      comm.cout0() << "\nDumping query results to " << opt.query_result_file_path
                   << std::endl;
      saltatlas::utility::gather_and_dump_neighbors(
          query_results, opt.query_result_file_path, comm);
    }
  }

  if (!opt.index_dump_prefix.empty()) {
    comm.cout0() << "\nDumping index to " << opt.index_dump_prefix << std::endl;
    g.dump_graph(opt.index_dump_prefix, opt.dump_index_with_distance);
    comm.cf_barrier();
    comm.cout0() << "Finished dumping." << std::endl;
  }

  return 0;
}

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.distance_name.clear();
  opt.point_file_names.clear();
  opt.point_file_format.clear();
  opt.index_dump_prefix.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:f:p:um:e:q:n:g:o:b:G:Dvh")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::stoi(optarg);
        break;

      case 'r':
        opt.r = std::stod(optarg);
        break;

      case 'd':
        opt.delta = std::stod(optarg);
        break;

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'u':
        opt.make_index_undirected = true;
        break;

      case 'm':
        opt.pruning_degree_multiplier = std::stod(optarg);
        break;

      case 'e':
        opt.epsilon = std::stold(optarg);
        break;

      case 'q':
        opt.query_file_path = optarg;
        break;

      case 'n':
        opt.query_k = std::stoi(optarg);
        break;

      case 'g':
        opt.ground_truth_file_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'b':
        opt.batch_size = std::stoul(optarg);
        break;

      case 'G':
        opt.index_dump_prefix = optarg;
        break;

      case 'D':
        opt.dump_index_with_distance = true;
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.distance_name.empty() || opt.point_file_format.empty() ||
      opt.point_file_names.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: " << exe_name << " [options] point_file1 [point_file2 ...]"
       << std::endl;
  cout << "Options:" << std::endl;
  cout << "  -k <int>    kNNG k parameter (required)" << std::endl;
  cout << "  -f <string> Distance name (required)" << std::endl;
  cout << "  -p <string> Point file format (required)" << std::endl;
  cout << "  -r <float> NN-Descent r parameter (default: 0.8)" << std::endl;
  cout << "  -d <float> NN-Descent delta parameter (default: 0.001)"
       << std::endl;
  cout << "  -u          Make index undirected (default: false)" << std::endl;
  cout << "  -m <float> High degree pruning parameter, must be >= 0 "
          "(default: 0.0, no "
          "prunning)"
       << std::endl;
  cout << "  -q <string> Query file path" << std::endl;
  cout << "  -n <int>    #of nearest neighbors to search" << std::endl;
  cout << "  -e <float> Query epsilon parameter (default: 0.1)" << std::endl;
  cout << "  -g <string> Ground truth file path" << std::endl;
  cout << "  -o <string> Query result file path" << std::endl;
  cout << "  -b <int>    Batch size (default: 1^25)" << std::endl;
  cout << "  -D <string> kNNG dump prefix" << std::endl;
  cout << "  -M          Dump index with distance" << std::endl;
  cout << "  -v          Verbose mode" << std::endl;
  cout << "  -h          Show this message" << std::endl;
}

template <typename cout_type>
void show_options(const option_t &opt, cout_type &cout) {
  cout << "Options:" << std::endl;
  cout << "  k: " << opt.index_k << std::endl;
  cout << "  r: " << opt.r << std::endl;
  cout << "  delta: " << opt.delta << std::endl;
  cout << "  distance name: " << opt.distance_name << std::endl;
  cout << "  point file format: " << opt.point_file_format << std::endl;
  cout << "  make index undirected: " << opt.make_index_undirected << std::endl;
  cout << "  pruning degree multiplier: " << opt.pruning_degree_multiplier
       << std::endl;
  cout << "  query file path: " << opt.query_file_path << std::endl;
  cout << "  ground truth file path: " << opt.ground_truth_file_path
       << std::endl;
  cout << "  query result file path: " << opt.query_result_file_path
       << std::endl;
  cout << "  query k: " << opt.query_k << std::endl;
  cout << "  epsilon: " << opt.epsilon << std::endl;
  cout << "  batch size: " << opt.batch_size << std::endl;
  cout << "  index dump prefix: " << opt.index_dump_prefix << std::endl;
  cout << "  dump index with distance: " << opt.dump_index_with_distance
       << std::endl;
  cout << "  verbose: " << opt.verbose << std::endl;
}