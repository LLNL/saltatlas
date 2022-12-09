// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "dnnd_example_common.hpp"

bool parse_options(int argc, char **argv, std::string &datastore_path,
                   std::string &original_datastore_path, int &query_k,
                   std::size_t &batch_size, std::string &query_file_path,
                   std::string &ground_truth_file_path,
                   std::string &query_result_file_path, bool &verbose,
                   bool &help);

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  std::string datastore_path;
  std::string original_datastore_path;
  int         query_k{4};
  std::size_t batch_size{0};
  std::string query_file_path;
  std::string ground_truth_file_path;
  std::string query_result_file_path;
  bool        verbose{false};
  bool        help{false};

  if (!parse_options(argc, argv, datastore_path, original_datastore_path,
                     query_k, batch_size, query_file_path,
                     ground_truth_file_path, query_result_file_path, verbose,
                     help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }

  if (!original_datastore_path.empty()) {
    if (dnnd_pm_type::copy(original_datastore_path, datastore_path)) {
      comm.cout0() << "\nTransferred index." << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    comm.cout0() << "<<Query>>" << std::endl;
    dnnd_pm_type dnnd(dnnd_pm_type::open_read_only, datastore_path, comm,
                      verbose);

    dnnd_pm_type::query_store_type queries;
    saltatlas::read_query(query_file_path, queries, comm);

    comm.cout0() << "Executing queries" << std::endl;
    ygm::timer step_timer;
    const auto query_results = dnnd.query_batch(queries, query_k, batch_size);
    comm.cf_barrier();
    comm.cout0() << "\nProcessing queries took (s)\t" << step_timer.elapsed()
                 << std::endl;

    if (!ground_truth_file_path.empty()) {
      show_query_recall_score(query_results, ground_truth_file_path, comm);
      show_query_recall_score_tied_distance(query_results,
                                            ground_truth_file_path, comm);
    }

    if (!query_result_file_path.empty()) {
      saltatlas::utility::gather_and_dump_neighbors(
          query_results, query_result_file_path, comm);
    }
  }
END_BLOCK:
  comm.cf_barrier();

  return 0;
}

inline bool parse_options(int argc, char **argv, std::string &datastore_path,
                          std::string &original_datastore_path, int &query_k,
                          std::size_t &batch_size, std::string &query_file_path,
                          std::string &ground_truth_file_path,
                          std::string &query_result_file_path, bool &verbose,
                          bool &help) {
  datastore_path.clear();
  original_datastore_path.clear();
  query_file_path.clear();
  ground_truth_file_path.clear();
  query_result_file_path.clear();

  int n;
  while ((n = ::getopt(argc, argv, "b:q:n:g:o:z:x:vh")) != -1) {
    switch (n) {
      case 'b':
        batch_size = std::stoul(optarg);
        break;

      case 'q':
        query_file_path = optarg;
        break;

      case 'n':
        query_k = std::stoi(optarg);
        break;

      case 'g':
        ground_truth_file_path = optarg;
        break;

      case 'z':
        datastore_path = optarg;
        break;

      case 'x':
        original_datastore_path = optarg;
        break;

      case 'o':
        query_result_file_path = optarg;
        break;

      case 'v':
        verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }

  if (datastore_path.empty() || query_file_path.empty()) {
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
       << "\n\t-n [int] Number of nearest neighbors to find for each query "
          "point."
       << "\n\t-o [string] Path to store query results."
       << "\n\t-g [string] Path to a query ground truth file."
       << "\n\t-x [string] If specified, transfer an already constructed index "
          "from this path to path 'z' at the beginning."
       << "\n\t-b [long int] Batch size for query (0 is the full batch mode)."
       << "\n\t-v If specified, turn on the verbose mode."
       << "\n\t-h Show this menu." << std::endl;
}