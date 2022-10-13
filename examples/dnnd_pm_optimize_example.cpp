// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include "dnnd_example_common.hpp"

bool parse_options(int argc, char **argv, std::string &original_datastore_path,
                   std::string &datastore_path,
                   std::string &datastore_transfer_path,
                   bool        &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   bool &verbose, bool &help);

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  std::string original_datastore_path;
  std::string datastore_path;
  std::string datastore_transfer_path;
  bool        make_index_undirected{false};
  double      pruning_degree_multiplier{1.5};
  bool        remove_long_paths{false};
  bool        verbose{false};
  bool        help{true};

  if (!parse_options(argc, argv, original_datastore_path, datastore_path,
                     datastore_transfer_path, make_index_undirected,
                     pruning_degree_multiplier, remove_long_paths, verbose,
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
      comm.cout0() << "\nTransferred index from " << original_datastore_path
                   << " to " << datastore_path << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    dnnd_pm_type dnnd(dnnd_pm_type::open, datastore_path, comm, verbose);
    comm.cout0() << "\n<<Index Optimization>>" << std::endl;
    ygm::timer optimization_timer;
    dnnd.optimize_index(make_index_undirected, pruning_degree_multiplier,
                        remove_long_paths);
    comm.cout0() << "\nIndex optimization took (s)\t"
                 << optimization_timer.elapsed() << std::endl;
  }

  if (!datastore_transfer_path.empty()) {
    comm.cout0() << "\nTransferring index data store " << datastore_path
                 << " to " << datastore_transfer_path << std::endl;
    if (!dnnd_pm_type::copy(datastore_path, datastore_transfer_path)) {
      comm.cerr0() << "\nFailed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  comm.cout0() << "\nThe index is ready for query." << std::endl;
  comm.cf_barrier();

  return 0;
}

bool parse_options(int argc, char **argv, std::string &original_datastore_path,
                   std::string &datastore_path,
                   std::string &datastore_transfer_path,
                   bool        &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   bool &verbose, bool &help) {
  original_datastore_path.clear();
  datastore_path.clear();
  datastore_transfer_path.clear();
  make_index_undirected = false;
  remove_long_paths     = false;
  verbose               = false;
  help                  = false;

  int n;
  while ((n = ::getopt(argc, argv, "x:z:o:um:lvh")) != -1) {
    switch (n) {
      case 'x':
        original_datastore_path = optarg;
        break;

      case 'z':
        datastore_path = optarg;
        break;

      case 'o':
        datastore_transfer_path = optarg;
        break;

      case 'u':
        make_index_undirected = true;
        break;

      case 'm':
        pruning_degree_multiplier = std::stod(optarg);
        break;

      case 'l':
        remove_long_paths = true;
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

  if (datastore_path.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: mpirun -n [#of processes] " << exe_name
       << " [options (see below)]" << std::endl;

  cout << "Options:"
       << "\n\t-z [string, required] Path to an index to modify."
       << "\n\t-u If specified, make the index undirected."
       << "\n\t-m [double] Pruning degree multiplier (m) in PyNNDescent."
       << "\n\t\tCut every points' neighbors more than 'k' x 'm' before the "
          "query."
       << "\n\t-l If specified, remove long paths."
       << "\n\t-x [string] If specified, transfer an already constructed index "
          "from this path to path 'z' at the beginning."
       << "\n\t-o [string] If specified, transfer the index to this path at "
          "the end."
       << "\n\t-v If specified, turn on the verbose mode."
       << "\n\t-h Show this menu." << std::endl;
}