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

#include <saltatlas/dnnd/data_reader.hpp>
#include "dnnd_example_common.hpp"

bool parse_options(int argc, char **argv, int &index_k, double &r,
                   double &delta, bool &exchange_reverse_neighbors,
                   std::size_t &batch_size, std::string &distance_metric_name,
                   std::vector<std::string> &point_file_names,
                   std::string &point_file_format, std::string &datastore_path,
                   std::string &datastore_transfer_path, bool &verbose,
                   bool &help);

template <typename cout_type>
void usage(std::string_view exe_name, cout_type &cout);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  int                      index_k{0};
  double                   r{0.8};
  double                   delta{0.001};
  bool                     exchange_reverse_neighbors{false};
  std::size_t              batch_size{0};
  std::string              distance_metric_name;
  std::vector<std::string> point_file_names;
  std::string              point_file_format;
  std::string              datastore_path;
  std::string              datastore_transfer_path;
  bool                     help{false};
  bool                     verbose{false};

  if (!parse_options(argc, argv, index_k, r, delta, exchange_reverse_neighbors,
                     batch_size, distance_metric_name, point_file_names,
                     point_file_format, datastore_path, datastore_transfer_path,
                     verbose, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    usage(argv[0], comm.cerr0());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (help) {
    usage(argv[0], comm.cout0());
    return 0;
  }

  {
    dnnd_pm_type dnnd(dnnd_pm_type::create, datastore_path,
                      distance_metric_name, comm, std::random_device{}(),
                      verbose);

    comm.cout0() << "\n<<Read Points>>" << std::endl;
    ygm::timer point_read_timer;
    saltatlas::read_points(point_file_names, point_file_format, verbose,
                           dnnd.get_point_store(), dnnd.get_point_partitioner(),
                           comm);
    comm.cout0() << "\nReading points took (s)\t" << point_read_timer.elapsed()
                 << std::endl;

    comm.cout0() << "\n<<Index Construction>>" << std::endl;
    ygm::timer const_timer;
    dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                         batch_size);
    comm.cout0() << "\nIndex construction took (s)\t" << const_timer.elapsed()
                 << std::endl;
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

inline bool parse_options(int argc, char **argv, int &index_k, double &r,
                          double &delta, bool &exchange_reverse_neighbors,
                          std::size_t              &batch_size,
                          std::string              &distance_metric_name,
                          std::vector<std::string> &point_file_names,
                          std::string              &point_file_format,
                          std::string              &datastore_path,
                          std::string &datastore_transfer_path, bool &verbose,
                          bool &help) {
  distance_metric_name.clear();
  point_file_names.clear();
  point_file_format.clear();
  datastore_path.clear();
  datastore_transfer_path.clear();
  exchange_reverse_neighbors = false;
  verbose                    = false;
  help                       = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:z:x:f:p:eb:vh")) != -1) {
    switch (n) {
      case 'k':
        index_k = std::stoi(optarg);
        break;

      case 'r':
        r = std::stod(optarg);
        break;

      case 'd':
        delta = std::stod(optarg);
        break;

      case 'e':
        exchange_reverse_neighbors = true;
        break;

      case 'f':
        distance_metric_name = optarg;
        break;

      case 'z':
        datastore_path = optarg;
        break;

      case 'x':
        datastore_transfer_path = optarg;
        break;

      case 'p':
        point_file_format = optarg;
        break;

      case 'b':
        batch_size = std::stoul(optarg);
        break;

      case 'v':
        verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    point_file_names.emplace_back(argv[index]);
  }

  if (datastore_path.empty() || distance_metric_name.empty() ||
      point_file_format.empty() || point_file_names.empty()) {
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
      << "\n\t\t'l2' (L2), 'cosine' (cosine similarity), or 'jaccard'"
         "(Jaccard index)."
      << "\n\t-p [string, required] Format of input point files:"
      << "\n\t\t'wsv' (whitespace-separated values),"
      << "\n\t\t'wsv-id' (WSV format and the first column is point ID),"
      << "\n\t\tor 'csv-id' (CSV format and the first column is point ID)."
      << "\n\t-k [int] Number of neighbors to have for each point in the index."
      << "\n\t-r [double] Sample rate parameter (ρ) in NN-Descent."
      << "\n\t-d [double] Precision parameter (δ) in NN-Descent."
      << "\n\t-e If specified, generate reverse neighbors globally during the "
         "index construction."
      << "\n"
      << "\n\t-x [string] If specified, transfer index to this path at the end."
      << "\n\t-b [long int] Batch size for the index construction (0 is the "
         "full batch mode)."
      << "\n\t-v If specified, turn on the verbose mode."
      << "\n\t-h Show this menu." << std::endl;
}