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

#include <saltatlas/dnnd/dnnd.hpp>
#include <saltatlas/dnnd/point_reader.hpp>
#include "dnnd_example_common.hpp"

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;
using dnnd_type = saltatlas::dnnd<id_type, feature_element_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

void parse_options(int argc, char **argv, int &index_k, int &query_k, double &r,
                   double &delta, bool &exchange_reverse_neighbors,
                   bool   &make_index_undirected,
                   double &pruning_degree_multiplier, bool &remove_long_paths,
                   std::size_t &batch_size, std::string &distance_metric_name,
                   std::vector<std::string> &point_file_names,
                   std::string              &query_file_name,
                   std::string &ground_truth_neighbor_ids_file_name,
                   std::string &point_file_format, std::string &out_file_prefix,
                   bool &verbose);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  int                      index_k{0};
  int                      query_k{0};
  double                   r{0.8};
  double                   delta{0.001};
  bool                     exchange_reverse_neighbors{false};
  bool                     make_index_undirected{false};
  double                   pruning_degree_multiplier{1.5};
  bool                     remove_long_paths{false};
  std::size_t              batch_size{0};
  std::string              distance_metric_name;
  std::vector<std::string> point_file_names;
  std::string              query_file_name;
  std::string              ground_truth_neighbor_ids_file_name;
  std::string              point_file_format;
  std::string              out_file_prefix;
  bool                     verbose{false};

  parse_options(argc, argv, index_k, query_k, r, delta,
                exchange_reverse_neighbors, make_index_undirected,
                pruning_degree_multiplier, remove_long_paths, batch_size,
                distance_metric_name, point_file_names, query_file_name,
                ground_truth_neighbor_ids_file_name, point_file_format,
                out_file_prefix, verbose);

  dnnd_type dnnd(distance_metric_name, comm, std::random_device{}(), verbose);
  comm.cf_barrier();

  {
    comm.cout0() << "<<Read Points>>" << std::endl;
    ygm::timer step_timer;
    saltatlas::read_points(point_file_names, point_file_format, verbose,
                           dnnd.get_point_store(), dnnd.get_point_partitioner(),
                           comm);
    comm.cout0() << "\nReading points took (s)\t" << step_timer.elapsed()
                 << std::endl;
  }

  {
    comm.cout0() << "<<Index Construction>>" << std::endl;
    ygm::timer step_timer;
    dnnd.construct_index(index_k, r, delta, exchange_reverse_neighbors,
                         batch_size);
    comm.cf_barrier();
    comm.cout0() << "\nIndex construction took (s)\t" << step_timer.elapsed()
                 << std::endl;
    if (!out_file_prefix.empty()) {
      comm.cout0() << "Dumping the index" << std::endl;
      dnnd.dump_index(out_file_prefix + "-index");
    }
  }

  {
    comm.cout0() << "\n<<Optimizing the index for query>>" << std::endl;
    ygm::timer step_timer;
    dnnd.optimize_index(make_index_undirected, pruning_degree_multiplier,
                        remove_long_paths);
    comm.cf_barrier();
    comm.cout0() << "\nIndex optimization took (s)\t" << step_timer.elapsed()
                 << std::endl;
    if (!out_file_prefix.empty()) {
      comm.cout0() << "Dumping the optimized index" << std::endl;
      dnnd.dump_index(out_file_prefix + "-optimized-index");
    }
  }

  if (!query_file_name.empty()) {
    comm.cout0() << "\n<<Query>>" << std::endl;

    comm.cout0() << "Reading queries" << std::endl;
    dnnd_type::query_point_store_type query_points;
    read_query<dnnd_type>(query_file_name, query_points);
    comm.cf_barrier();

    comm.cout0() << "Executing queries" << std::endl;
    ygm::timer step_timer;
    const auto query_result =
        dnnd.query_batch(query_points, query_k, batch_size);
    comm.cf_barrier();
    comm.cout0() << "\nProcessing queries took (s)\t" << step_timer.elapsed()
                 << std::endl;

    if (!ground_truth_neighbor_ids_file_name.empty() ||
        !out_file_prefix.empty()) {
      const auto all_query_result =
          gather_query_result<neighbor_type>(query_result, comm);
      if (!ground_truth_neighbor_ids_file_name.empty() && comm.rank0()) {
        const auto ground_truth_neighbors =
            read_neighbor_ids<id_type>(ground_truth_neighbor_ids_file_name);
        show_accuracy(ground_truth_neighbors, all_query_result);
      }
      comm.cf_barrier();

      if (!out_file_prefix.empty()) {
        comm.cout0() << "\nDump query results" << std::endl;
        dump_query_result(all_query_result, out_file_prefix + "-query", comm);
      }
    }
  }
  comm.cf_barrier();

  return 0;
}

inline void parse_options(
    int argc, char **argv, int &index_k, int &query_k, double &r, double &delta,
    bool &exchange_reverse_neighbors, bool &make_index_undirected,
    double &pruning_degree_multiplier, bool &remove_long_paths,
    std::size_t &batch_size, std::string &distance_metric_name,
    std::vector<std::string> &point_file_names, std::string &query_file_name,
    std::string &ground_truth_neighbor_ids_file_name,
    std::string &point_file_format, std::string &out_file_prefix,
    bool &verbose) {
  distance_metric_name.clear();
  point_file_names.clear();
  point_file_format.clear();
  out_file_prefix.clear();
  exchange_reverse_neighbors = false;
  make_index_undirected      = false;
  remove_long_paths          = false;
  verbose                    = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:d:o:f:p:eb:vq:n:g:m:ul")) != -1) {
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

      case 'o':
        out_file_prefix = optarg;
        break;

      case 'p':
        point_file_format = optarg;
        break;

      case 'b':
        batch_size = std::stoul(optarg);
        break;

      case 'q':
        query_file_name = optarg;
        break;

      case 'n':
        query_k = std::stoi(optarg);
        break;

      case 'g':
        ground_truth_neighbor_ids_file_name = optarg;
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
