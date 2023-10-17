// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "dnnd_example_common.hpp"

struct option_t {
  std::string datastore_path;
  std::string original_datastore_path;
  int         query_k{4};
  double      epsilon{0.1};
  double      mu{0.0};
  std::size_t batch_size{0};
  std::string query_file_path;
  std::string ground_truth_file_path;
  std::string query_result_file_path;
  bool        verbose{true};
};

bool parse_options(int, char **, option_t &, bool &);
template <typename cout_type>
void usage(std::string_view, cout_type &);
void show_options(const option_t &, ygm::comm &);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  show_config(comm);

  bool     help{false};
  option_t opt;
  if (!parse_options(argc, argv, opt, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    return 0;
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }
  show_options(opt, comm);

  if (!opt.original_datastore_path.empty()) {
    if (dnnd_pm_type::copy(opt.original_datastore_path, opt.datastore_path)) {
      comm.cout0() << "\nTransferred index." << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    comm.cout0() << "\n<<Query>>" << std::endl;
    dnnd_pm_type dnnd(dnnd_pm_type::open_read_only, opt.datastore_path, comm,
                      opt.verbose);

    dnnd_pm_type::query_store_type queries;
    saltatlas::read_query(opt.query_file_path, queries, comm);

    comm.cout0() << "Executing queries" << std::endl;
    ygm::timer step_timer;
    const auto query_results = dnnd.query_batch(
        queries, opt.query_k, opt.epsilon, opt.mu, opt.batch_size);
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
      saltatlas::utility::gather_and_dump_neighbors(
          query_results, opt.query_result_file_path, comm);
    }
  }
END_BLOCK:
  comm.cf_barrier();

  return 0;
}

inline bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.datastore_path.clear();
  opt.original_datastore_path.clear();
  opt.query_file_path.clear();
  opt.ground_truth_file_path.clear();
  opt.query_result_file_path.clear();

  int n;
  while ((n = ::getopt(argc, argv, "b:q:n:g:o:z:x:e:m:vh")) != -1) {
    switch (n) {
      case 'b':
        opt.batch_size = std::stoul(optarg);
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

      case 'z':
        opt.datastore_path = optarg;
        break;

      case 'x':
        opt.original_datastore_path = optarg;
        break;

      case 'o':
        opt.query_result_file_path = optarg;
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'e':
        opt.epsilon = std::stold(optarg);
        break;

      case 'm':
        opt.mu = std::stold(optarg);
        break;

      case 'h':
        help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }

  if (opt.datastore_path.empty() || opt.query_file_path.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: mpirun -n [#of processes] " << exe_name
       << " [options (see below)]" << std::endl;

  cout << "Options:"
       << "\n\t-z [string, required] Path to an index."
       << "\n\t-q [string, required] Path to a query file."
       << "\n\t-n [int, required] Number of nearest neighbors to find for each "
          "query point."
       << "\n\t-e [double] Epsilon parameter in PyNNDescent."
       << "\n\t-o [string] Path to store query results."
       << "\n\t-g [string] Path to a query ground truth file."
       << "\n\t-x [string] If specified, transfer an already constructed index "
          "from this path to path 'z' at the beginning."
       << "\n\t-b [long int] Batch size for query (0 is the full batch mode)."
       << "\n\t-v If specified, turn on the verbose mode."
       << "\n\t-h Show this menu." << std::endl;
}

void show_options(const option_t &opt, ygm::comm &comm) {
  comm.cout0() << "Options:"
               << "\nOriginal datastore path\t" << opt.original_datastore_path
               << "\nDatastore path\t" << opt.datastore_path
               << "\nQuery file path\t" << opt.query_file_path
               << "\nQuery n (#of neighbors to search)\t" << opt.query_k
               << "\nEpsilon\t" << opt.epsilon << "\nMu\t" << opt.mu
               << "\nBatch size\t" << opt.batch_size
               << "\nGround truth file path\t" << opt.ground_truth_file_path
               << "\nQuery result file path\t" << opt.query_result_file_path
               << "\nVerbose\t" << opt.verbose << std::endl;
}