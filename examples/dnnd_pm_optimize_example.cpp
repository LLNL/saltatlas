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

struct option_t {
  std::string original_datastore_path;
  std::string datastore_path;
  std::string datastore_transfer_path;
  bool        make_index_undirected{true};
  double      pruning_degree_multiplier{0.0};  // no pruning by default
  bool        remove_long_paths{false};
  std::size_t batch_size{1ULL << 28};
  std::string index_dump_prefix{false};
  bool        verbose{true};
};

bool parse_options(int, char **, option_t &, bool &);
template <typename cout_type>
void usage(std::string_view, cout_type &);
void show_options(const option_t &, ygm::comm &);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  show_config(comm);

  option_t opt;
  bool     help{true};

  if (!parse_options(argc, argv, opt, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }

  show_options(opt, comm);

  if (!opt.original_datastore_path.empty()) {
    if (dnnd_pm_type::copy(opt.original_datastore_path, opt.datastore_path)) {
      comm.cout0() << "\nTransferred index from " << opt.original_datastore_path
                   << " to " << opt.datastore_path << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    dnnd_pm_type dnnd(dnnd_pm_type::open, opt.datastore_path, comm,
                      opt.verbose);
    comm.cout0() << "\n<<Index Optimization>>" << std::endl;
    ygm::timer optimization_timer;
    dnnd.optimize_index(opt.make_index_undirected,
                        opt.pruning_degree_multiplier, opt.remove_long_paths);
    comm.cout0() << "\nIndex optimization took (s)\t"
                 << optimization_timer.elapsed() << std::endl;
  }
  comm.cout0() << "\nThe index is ready for query." << std::endl;

  if (!opt.datastore_transfer_path.empty()) {
    comm.cout0() << "\nTransferring index data store " << opt.datastore_path
                 << " to " << opt.datastore_transfer_path << std::endl;
    if (!dnnd_pm_type::copy(opt.datastore_path, opt.datastore_transfer_path)) {
      comm.cerr0() << "\nFailed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  if (!opt.index_dump_prefix.empty()) {
    comm.cout0() << "\nDumping index to " << opt.index_dump_prefix << std::endl;
    // Reopen dnnd in read-only mode
    dnnd_pm_type dnnd(dnnd_pm_type::open_read_only, opt.datastore_path, comm,
                      opt.verbose);
    if (!dnnd.dump_index(opt.index_dump_prefix)) {
      comm.cerr0() << "\nFailed to dump index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    comm.cf_barrier();
    comm.cout0() << "Finished dumping." << std::endl;
  }

  return 0;
}

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.original_datastore_path.clear();
  opt.datastore_path.clear();
  opt.datastore_transfer_path.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "i:z:x:um:lb:D:vh")) != -1) {
    switch (n) {
      case 'i':
        opt.original_datastore_path = optarg;
        break;

      case 'z':
        opt.datastore_path = optarg;
        break;

      case 'x':
        opt.datastore_transfer_path = optarg;
        break;

      case 'u':
        opt.make_index_undirected = true;
        break;

      case 'm':
        opt.pruning_degree_multiplier = std::stod(optarg);
        break;

      case 'l':
        opt.remove_long_paths = true;
        break;

      case 'b':
        opt.batch_size = std::stoull(optarg);
        break;

      case 'D':
        opt.index_dump_prefix = optarg;
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }

  if (opt.datastore_path.empty()) {
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
       << "\n\t\tCut every points' neighbors more than 'k' x 'm'"
          "No pruning if <= 0."
       << "\n\t-l If specified, remove long paths as proposed by PyNNDescent."
       << "\n\t-i [string] If specified, transfer an already constructed index "
          "from this path to path 'z' at the beginning."
       << "\n\t-x [string] If specified, transfer the index to this path at "
          "the end."
       << "\n\t-b [long int] Batch size (0 is the full batch mode)."
       << "\n\t-D [string] If specified, dump the k-NN index (only neighbor "
          "IDs) to files starting with this prefix at the end (one file per "
          "process)."
       << "\n"
       << "\n\t-v If specified, turn on the verbose mode."
       << "\n\t-h Show this menu." << std::endl;
}

void show_options(const option_t &opt, ygm::comm &comm) {
  comm.cout0() << "\nOptions:"
               << "\nOriginal datastore path\t" << opt.original_datastore_path
               << "\nDatastore path\t" << opt.datastore_path
               << "\nMake index undirected\t" << opt.make_index_undirected
               << "\nPruning degree multiplier\t"
               << opt.pruning_degree_multiplier << "\nRemove long paths\t"
               << opt.remove_long_paths << "\nBatch size\t" << opt.batch_size
               << "\nDatastore transfer path\t" << opt.datastore_transfer_path
               << "\nk-NN index dump file prefix\t" << opt.index_dump_prefix
               << "\nVerbose\t" << opt.verbose << std::endl;
}