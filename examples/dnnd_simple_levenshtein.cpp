// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT
/*
Usage example:
cd saltatlas/build
mpirun -n 2 ./examples/dnnd_simple_levenshtein -p str -u  -k 3 -n 4\
  -q ../examples/datasets/query_string.txt \
  -g ../examples/datasets/ground-truth_string.txt \
  ../examples/datasets/point_string.txt
*/

#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_simple.hpp>
#include <saltatlas/dnnd/utility.hpp>

#include "dnnd_example_common.hpp"

using id_type = uint32_t;
using dist_t  = int;  // Levenshtein distance is an integer
// Use char to represent a string
using point_type = saltatlas::feature_vector<char>;

struct option_t {
  // KNNG construction options
  int         index_k{2};
  double      r{0.8};
  double      delta{0.001};
  std::size_t batch_size{1ULL << 25};

  // KNNG optimization options
  bool   make_index_undirected{false};
  double pruning_degree_multiplier{0.0};

  // Query options
  int    query_k{1};  // #of neighbors to search for
  double epsilon{0.1};

  // Data dump options
  std::filesystem::path index_dump_prefix;
  bool                  dump_index_with_distance{false};
  bool                  verbose{false};

  // Input file arguments
  std::vector<std::filesystem::path> point_file_paths;
  std::filesystem::path              query_file_path;
  std::filesystem::path              ground_truth_file_path;
  std::filesystem::path              query_result_file_path;
  // 'str' is for files that contain strings (one string per line)
  // 'str-id' is for files that contain IDs in the first column
  std::string point_file_format;
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help);
template <typename cout_type>
void usage(std::string_view, cout_type &);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

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

  saltatlas::dnnd<id_t, point_type, dist_t> g(
      saltatlas::distance::id::levenshtein, comm);

  comm.cout0() << "<<Read Points>>" << std::endl;
  // Read string points, where each line in files is a string
  g.load_points(opt.point_file_paths.begin(), opt.point_file_paths.end(),
                opt.point_file_format);

  comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
  g.build(opt.index_k, opt.r, opt.delta, opt.batch_size);

  if (opt.make_index_undirected) {
    comm.cout0() << "\n<<kNNG Optimization>>" << std::endl;
    g.optimize(opt.make_index_undirected, opt.pruning_degree_multiplier);
  }

  if (!opt.query_file_path.empty() && opt.query_k > 0) {
    comm.cout0() << "\n<<Query>>" << std::endl;
    std::vector<point_type> queries;
    saltatlas::read_query(opt.query_file_path, queries, comm);

    comm.cout0() << "Executing queries" << std::endl;
    const auto query_results =
        g.query(queries.begin(), queries.end(), opt.query_k, opt.epsilon);
    comm.cf_barrier();

    if (!opt.ground_truth_file_path.empty()) {
      show_query_recall_score(query_results, opt.ground_truth_file_path, comm);
    }

    if (!opt.query_result_file_path.empty()) {
      comm.cout0() << "\nDumping query results to "
                   << opt.query_result_file_path << std::endl;
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

  if (optind < argc) {
    opt.point_file_paths.clear();
    for (int index = optind; index < argc; index++) {
      opt.point_file_paths.emplace_back(argv[index]);
    }
  }

  if (opt.point_file_format.empty() || opt.point_file_paths.empty()) {
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
  cout << "  -p <string> Point file format (required). wsv, wsv-id, csv, "
          "csv-id, str, and str-id are supported"
       << std::endl;
  cout << "  -r <float>  NN-Descent r parameter (default: 0.8)" << std::endl;
  cout << "  -d <float>  NN-Descent delta parameter (default: 0.001)"
       << std::endl;
  cout << "  -u          Make index undirected (default: false)" << std::endl;
  cout << "  -m <float>  High degree pruning parameter, must be >= 0 "
          "(default: 0.0, no prunning)"
       << std::endl;
  cout << "  -q <string> Query file path" << std::endl;
  cout << "  -n <int>    #of nearest neighbors to search" << std::endl;
  cout << "  -e <float>  Query epsilon parameter (default: 0.1)" << std::endl;
  cout << "  -g <string> Ground truth file path" << std::endl;
  cout << "  -o <string> Query result file path" << std::endl;
  cout << "  -b <int>    Batch size (default: 1^25)" << std::endl;
  cout << "  -G <string> kNNG dump prefix" << std::endl;
  cout << "  -D          Dump index with distance" << std::endl;
  cout << "  -v          Verbose mode" << std::endl;
  cout << "  -h          Show this message" << std::endl;
}