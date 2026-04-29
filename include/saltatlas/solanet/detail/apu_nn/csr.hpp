// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include "saltatlas/solanet/detail/apu_nn/memory.hpp"

namespace saltatlas::solanet::apu_nn {

template <typename id_type, typename dist_type>
struct csr_graph {
  csr_graph() = default;
  csr_graph(const size_t n_vertices, const size_t n_edges)
      : offsets(allocate_hip_memory<id_type>(n_vertices + 1)),
        ids(allocate_hip_memory<id_type>(n_edges)),
        distances(allocate_hip_memory<dist_type>(n_edges)) {}

  hip_unique_ptr<id_type>   offsets;
  hip_unique_ptr<id_type>   ids;
  hip_unique_ptr<dist_type> distances;
};
}  // namespace saltatlas::solanet::apu_nn