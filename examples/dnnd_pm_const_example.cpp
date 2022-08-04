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

void parse_options(int argc, char **argv, int &index_k, double &r,
                   double &delta, bool &exchange_reverse_neighbors,
                   std::size_t &batch_size, std::string &distance_metric_name,
                   std::vector<std::string> &point_file_names,
                   std::string &point_file_format, std::string &datastore_path,
                   std::string &datastore_transfer_path,
                   bool        &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   bool &verbose);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);
  {
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
    bool                     make_index_undirected{false};
    double                   pruning_degree_multiplier{1.5};
    bool                     remove_long_paths{false};
    bool                     verbose{false};

    parse_options(argc, argv, index_k, r, delta, exchange_reverse_neighbors,
                  batch_size, distance_metric_name, point_file_names,
                  point_file_format, datastore_path, datastore_transfer_path,
                  make_index_undirected, pruning_degree_multiplier,
                  remove_long_paths, verbose);

    {
      dnnd_type dnnd(dnnd_type::create, datastore_path, distance_metric_name,
                     comm, std::random_device{}(), verbose);

      comm.cout0() << "\n<<Read Points>>" << std::endl;
      ygm::timer point_read_timer;
      saltatlas::read_points(point_file_names, point_file_format, verbose,
                             dnnd.get_point_store(),
                             dnnd.get_point_partitioner(), comm);
      comm.cout0() << "\nReading points took (s)\t"
                   << point_read_timer.elapsed() << std::endl;

      comm.cout0() << "\n<<Index Construction>>" << std::endl;
      ygm::timer const_timer;
      dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                           batch_size);
      comm.cout0() << "\nIndex construction took (s)\t" << const_timer.elapsed()
                   << std::endl;

      comm.cout0() << "\n<<Index Optimization>>" << std::endl;
      ygm::timer optimization_timer;
      dnnd.optimize_index(make_index_undirected, pruning_degree_multiplier,
                          remove_long_paths);
      comm.cout0() << "\nIndex optimization took (s)\t"
                   << optimization_timer.elapsed() << std::endl;
    }
    comm.cout0() << "\nThe index is ready for query." << std::endl;

    if (!datastore_transfer_path.empty()) {
      if (dnnd_type::copy(datastore_path, datastore_transfer_path)) {
        comm.cout0() << "\nTransferred index data store." << std::endl;
      } else {
        comm.cerr0() << "\nFailed to transfer index." << std::endl;
      }
    }
  }

  return 0;
}

inline void parse_options(
    int argc, char **argv, int &index_k, double &r, double &delta,
    bool &exchange_reverse_neighbors, std::size_t &batch_size,
    std::string              &distance_metric_name,
    std::vector<std::string> &point_file_names, std::string &point_file_format,
    std::string &datastore_path, std::string &datastore_transfer_path,
    bool &make_index_undirected, double &pruning_degree_multiplier,
    bool &remove_long_paths, bool &verbose) {
  distance_metric_name.clear();
  point_file_names.clear();
  point_file_format.clear();
  datastore_path.clear();
  datastore_transfer_path.clear();
  exchange_reverse_neighbors = false;
  verbose                    = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:z:x:f:p:eb:um:lv")) != -1) {
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

  for (int index = optind; index < argc; index++) {
    point_file_names.emplace_back(argv[index]);
  }
}