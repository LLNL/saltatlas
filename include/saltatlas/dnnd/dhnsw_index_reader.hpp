// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/io/ndjson_parser.hpp>

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>

namespace saltatlas {

template <typename id_type>
inline void read_dhnsw_index(
    const std::vector<std::string> &index_file_names, const bool verbose,
    const std::function<int(const id_type &id)>       &point_partitioner,
    std::unordered_map<id_type, std::vector<id_type>> &local_store,
    ygm::comm                                         &comm) {
  std::size_t num_invalid_lines = 0;

  ygm::ygm_ptr<std::unordered_map<id_type, std::vector<id_type>>> ptr_store(
      &local_store);

  auto process_base_json = [&verbose, &point_partitioner, &num_invalid_lines,
                            &comm, &ptr_store](auto line) {
    if (!line.contains("ID")) {
      ++num_invalid_lines;
      return;
    }
    const auto id = static_cast<id_type>(line.at("ID").as_int64());

    if (!line.contains("Neighbors") || !line.at("Neighbors").is_array()) {
      ++num_invalid_lines;
      return;
    }
    std::vector<id_type> neighbors;
    for (auto &item : line.at("Neighbors").as_array()) {
      const auto nghbr = item.as_int64();
      neighbors.push_back(nghbr);
    }

    // Send to the corresponding rank
    auto receiver = [](auto, auto store, const id_t id, const auto &neighbors) {
      if (store->count(id)) {
        std::cerr << "Duplicate ID " << id << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      (*store)[id] = std::move(neighbors);
    };
    comm.async(point_partitioner(id), receiver, ptr_store, id, neighbors);
  };

  ygm::io::ndjson_parser jsonp(comm, index_file_names, false, false);
  jsonp.for_all(process_base_json);
  comm.barrier();

  comm.cout0() << "#of invalid line\t" << comm.all_reduce_sum(num_invalid_lines)
               << std::endl;
}

}  // namespace saltatlas