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
#include "dnnd_example_common.hpp"

using id_type              = uint32_t;
using feature_element_type = float;
using distance_type        = float;
using dnnd_type =
    saltatlas::dnnd_pm<id_type, feature_element_type, distance_type>;
using neighbor_type = typename dnnd_type::neighbor_type;

static constexpr std::size_t k_ygm_buff_size = 256 * 1024 * 1024;

void parse_options(int argc, char **argv, std::string &datastore_path,
                   std::string &original_datastore_path, int &query_k,
                   std::size_t &batch_size, std::string &query_file_name,
                   std::string &ground_truth_neighbor_ids_file_name,
                   std::string &query_result_file_path, bool &verbose);

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv, k_ygm_buff_size);

  std::string datastore_path;
  std::string original_datastore_path;
  int         query_k{0};
  std::size_t batch_size{0};
  std::string query_file_name;
  std::string ground_truth_neighbor_ids_file_name;
  std::string query_result_file_name;
  bool        verbose{false};

  parse_options(argc, argv, datastore_path, original_datastore_path, query_k,
                batch_size, query_file_name,
                ground_truth_neighbor_ids_file_name, query_result_file_name,
                verbose);

  if (!original_datastore_path.empty()) {
    if (dnnd_type::copy(original_datastore_path, datastore_path)) {
      comm.cout0() << "\nTransferred index." << std::endl;
    } else {
      comm.cerr0() << "Failed to transfer index." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  {
    dnnd_type dnnd(dnnd_type::open_read_only, datastore_path, comm, verbose);
    comm.cout0() << "<<Query>>" << std::endl;

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
        !query_result_file_name.empty()) {
      const auto all_query_result =
          gather_query_result<neighbor_type>(query_result, comm);
      if (!ground_truth_neighbor_ids_file_name.empty() && comm.rank0()) {
        const auto ground_truth_neighbors =
            read_neighbor_ids<id_type>(ground_truth_neighbor_ids_file_name);
        show_accuracy(ground_truth_neighbors, all_query_result);
      }
      comm.cf_barrier();

      if (!query_result_file_name.empty()) {
        comm.cout0() << "\nDumping query results" << std::endl;
        dump_query_result(all_query_result, query_result_file_name, comm);
      }
    }
  }

  comm.cf_barrier();

  return 0;
}

inline void parse_options(int argc, char **argv, std::string &datastore_path,
                          std::string &original_datastore_path, int &query_k,
                          std::size_t &batch_size, std::string &query_file_name,
                          std::string &ground_truth_neighbor_ids_file_name,
                          std::string &query_result_file_path, bool &verbose) {
  datastore_path.clear();
  original_datastore_path.clear();
  query_file_name.clear();
  ground_truth_neighbor_ids_file_name.clear();
  query_result_file_path.clear();

  int n;
  while ((n = ::getopt(argc, argv, "b:q:n:g:o:z:x:v")) != -1) {
    switch (n) {
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

      default:
        std::cerr << "Invalid option" << std::endl;
        std::abort();
    }
  }
}
