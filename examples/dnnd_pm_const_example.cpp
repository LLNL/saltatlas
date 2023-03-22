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
  int                      index_k{0};
  double                   r{0.8};
  double                   delta{0.001};
  bool                     exchange_reverse_neighbors{true};
  std::size_t              batch_size{1ULL << 31};
  std::string              distance_metric_name;
  std::vector<std::string> point_file_names;
  std::string              point_file_format;
  std::string              dnnd_init_index_path;
  std::string              dhnsw_init_index_path;
  std::string              datastore_path;
  std::string              datastore_transfer_path;
  bool                     verbose{false};
};

bool parse_options(int argc, char **argv, option_t &option, bool &help);

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  show_config(comm);

  option_t opt;
  bool     help{false};
  if (!parse_options(argc, argv, opt, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }

  {
    dnnd_pm_type dnnd(dnnd_pm_type::create, opt.datastore_path,
                      opt.distance_metric_name, comm, std::random_device{}(),
                      opt.verbose);

    comm.cout0() << "\n<<Read Points>>" << std::endl;
    ygm::timer point_read_timer;
    saltatlas::read_points(opt.point_file_names, opt.point_file_format,
                           opt.verbose, dnnd.get_point_partitioner(),
                           dnnd.get_point_store(), comm);
    comm.cout0() << "\nReading points took (s)\t" << point_read_timer.elapsed()
                 << std::endl;

    comm.cout0() << "\n<<Index Construction>>" << std::endl;
    ygm::timer const_timer;
    if (!opt.dnnd_init_index_path.empty()) {
      dnnd_pm_type init_dnnd(dnnd_pm_type::open_read_only,
                             opt.dnnd_init_index_path, comm, opt.verbose);
      dnnd.construct_index(opt.index_k, opt.r, opt.delta,
                           opt.exchange_reverse_neighbors, opt.batch_size,
                           init_dnnd.get_knn_index());
    } else if (!opt.dhnsw_init_index_path.empty()) {
      std::unordered_map<id_type, std::vector<id_type>> init_neighbors;
      comm.cout0() << "Read DHNS index" << std::endl;
      ygm::timer read_timer;
      saltatlas::read_dhnsw_index(
          std::vector<std::string>{opt.dhnsw_init_index_path}, opt.verbose,
          dnnd.get_point_partitioner(), init_neighbors, comm);
      comm.cout0() << "\nReading index took (s)\t" << read_timer.elapsed()
                   << std::endl;

      dnnd.construct_index(opt.index_k, opt.r, opt.delta,
                           opt.exchange_reverse_neighbors, opt.batch_size,
                           init_neighbors);
    } else {
      dnnd.construct_index(opt.index_k, opt.r, opt.delta,
                           opt.exchange_reverse_neighbors, opt.batch_size);
    }
    comm.cout0() << "\nIndex construction took (s)\t" << const_timer.elapsed()
                 << std::endl;
  }
  comm.cf_barrier();
  comm.cout0() << "\nClosed Metall." << std::endl;

  if (!opt.datastore_transfer_path.empty()) {
    comm.cout0() << "\nTransferring index data store " << opt.datastore_path
                 << " to " << opt.datastore_transfer_path << std::endl;
    if (!dnnd_pm_type::copy(opt.datastore_path, opt.datastore_transfer_path)) {
      comm.cerr0() << "\nFailed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    comm.cout0() << "Finished transfer." << std::endl;
  }

  return 0;
}

inline bool parse_options(int argc, char **argv, option_t &option, bool &help) {
  option.distance_metric_name.clear();
  option.point_file_names.clear();
  option.point_file_format.clear();
  option.dnnd_init_index_path.clear();
  option.dhnsw_init_index_path.clear();
  option.datastore_path.clear();
  option.datastore_transfer_path.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:z:x:f:p:I:H:eb:vh")) != -1) {
    switch (n) {
      case 'k':
        option.index_k = std::stoi(optarg);
        break;

      case 'r':
        option.r = std::stod(optarg);
        break;

      case 'd':
        option.delta = std::stod(optarg);
        break;

      case 'e':
        option.exchange_reverse_neighbors = true;
        break;

      case 'f':
        option.distance_metric_name = optarg;
        break;

      case 'z':
        option.datastore_path = optarg;
        break;

      case 'x':
        option.datastore_transfer_path = optarg;
        break;

      case 'p':
        option.point_file_format = optarg;
        break;

      case 'I':
        option.dnnd_init_index_path = optarg;
        break;

      case 'H':
        option.dhnsw_init_index_path = optarg;
        break;

      case 'b':
        option.batch_size = std::stoul(optarg);
        break;

      case 'v':
        option.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    option.point_file_names.emplace_back(argv[index]);
  }

  if (option.datastore_path.empty() || option.distance_metric_name.empty() ||
      option.point_file_format.empty() || option.point_file_names.empty()) {
    return false;
  }

  if (!option.dnnd_init_index_path.empty() &&
      !option.dhnsw_init_index_path.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: mpirun -n [#of processes] " << exe_name
       << " [options (see below)] [list of input point files (required)]"
       << std::endl;

  cout
      << "Options:"
      << "\n\t-z [string, required] Path to store constructed index."
      << "\n\t-f [string, required] Distance metric name:"
      << "\n\t\t'l2' (L2), sql2' (squared L2), 'cosine' (cosine similarity), "
         "or 'jaccard' (Jaccard index)."
      << "\n\t-p [string, required] Format of input point files:"
      << "\n\t\t'wsv' (whitespace-separated values w/o ID),"
      << "\n\t\t'wsv-id' (WSV format and the first column is point ID),"
      << "\n\t\t'csv' (comma-separated values w/o ID),"
      << "\n\t\tor 'csv-id' (CSV format and the first column is point ID)."
      << "\n\t-k [int] Number of neighbors to have for each point in the index."
      << "\n\t-r [double] Sample rate parameter (ρ) in NN-Descent."
      << "\n\t-d [double] Precision parameter (δ) in NN-Descent."
      << "\n\t-e If specified, generate reverse neighbors globally during the "
         "index construction."
      << "\n"
      << "\n\t-I [string] Path to an existing DNND data for initializing the "
         "new index."
      << "\n\t-H [string] Path to an existing HNSW index directory for"
         " initializing the new index."
      << "\n"
      << "\n\t-x [string] If specified, transfer index to this path at the end."
      << "\n\t-b [long int] Batch size for the index construction (0 is the "
         "full batch mode)."
      << "\n\t-v If specified, turn on the verbose mode."
      << "\n\t-h Show this menu." << std::endl;
}