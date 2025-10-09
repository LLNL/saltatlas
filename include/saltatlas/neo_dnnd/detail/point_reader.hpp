// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include "saltatlas/common/data_reader.hpp"
#include "saltatlas/common/point_store.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"

namespace saltatlas::dndetail {

// Wrapper function to use the common read-points function
template <typename id_type, typename fe_type>
static std::pair<std::vector<id_type>, std::vector<std::vector<fe_type>>>
read_points(const std::vector<std::filesystem::path>&    paths,
            const std::string&                           format,
            const std::function<int(const id_type& id)>& partitioner,
            const bool verbose, saltatlas::mpi::communicator& comm) {
  point_store<id_type, std::vector<fe_type>> pstore;
  {
    ygm::comm ygm_comm(comm.comm());
    saltatlas::read_points(paths, format, verbose, partitioner, pstore,
                           ygm_comm);
  }

  std::vector<id_type>              ids;
  std::vector<std::vector<fe_type>> fvs;
  ids.reserve(pstore.size());
  fvs.reserve(pstore.size());
  for (auto& [id, fv] : pstore) {
    ids.push_back(id);
    fvs.push_back(std::move(fv));
  }
  return std::make_pair(std::move(ids), std::move(fvs));
}
}  // namespace saltatlas::dndetail