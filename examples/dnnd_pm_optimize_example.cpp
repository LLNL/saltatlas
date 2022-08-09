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

#include <saltatlas/dnnd/dnnd_pm.hpp>
#include <saltatlas/dnnd/point_reader.hpp>
#include "dnnd_example_common.hpp"

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;
using dnnd_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

void parse_options(int argc, char **argv, std::string &original_datastore_path,
                   std::string &datastore_path,
                   std::string &datastore_transfer_path,
                   bool        &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   bool &verbose);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  std::string original_datastore_path;
  std::string datastore_path;
  std::string datastore_transfer_path;
  bool        make_index_undirected{false};
  double      pruning_degree_multiplier{1.5};
  bool        remove_long_paths{false};
  bool        verbose{false};

  parse_options(argc, argv, original_datastore_path, datastore_path,
                datastore_transfer_path, make_index_undirected,
                pruning_degree_multiplier, remove_long_paths, verbose);

  if (!original_datastore_path.empty()) {
    if (dnnd_type::copy(original_datastore_path, datastore_path)) {
      comm.cout0() << "\nTransferred index from " << original_datastore_path
                   << " to " << datastore_path << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    dnnd_type dnnd(dnnd_type::open, datastore_path, comm, verbose);
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
    if (!dnnd_type::copy(datastore_path, datastore_transfer_path)) {
      comm.cerr0() << "\nFailed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  comm.cout0() << "\nThe index is ready for query." << std::endl;

  return 0;
}

void parse_options(int argc, char **argv, std::string &original_datastore_path,
                   std::string &datastore_path,
                   std::string &datastore_transfer_path,
                   bool        &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   bool &verbose) {
  original_datastore_path.clear();
  datastore_path.clear();
  datastore_transfer_path.clear();

  int n;
  while ((n = ::getopt(argc, argv, "x:z:o:um:lv")) != -1) {
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

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }
}