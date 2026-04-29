// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

#include "saltatlas/solanet/detail/apu_nn/csr.hpp"
#include "saltatlas/solanet/detail/apu_nn/matrix.hpp"
#include "saltatlas/solanet/detail/apu_nn/memory.hpp"
#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {
namespace detail {

// Returns true when an ID refers to a valid local vertex index.
template <typename id_type>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE bool is_valid_vertex_id(
    const id_type id, const size_t n_vertices) {
  if constexpr (std::is_signed_v<id_type>) {
    if (id < 0) {
      return false;
    }
  }
  return static_cast<size_t>(id) < n_vertices;
}

// Counts incoming edges per destination vertex from dense KNNG rows.
template <typename id_type>
SALTATLAS_HD_GLOBAL void count_reverse_edges_kernel(
    const matrix_view<const id_type> in_knng_ids, uint64_t* in_degree_counts) {
  const size_t total_edges = in_knng_ids.size();
  const size_t stride      = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t       idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  while (idx < total_edges) {
    const id_type nid = in_knng_ids.data()[idx];
    if (is_valid_vertex_id(nid, in_knng_ids.n_rows())) {
      atomicAdd(&in_degree_counts[static_cast<size_t>(nid)], 1ULL);
    }
    idx += stride;
  }
}

// Clips per-vertex reverse degree to the requested maximum.
SALTATLAS_HD_GLOBAL void clip_counts_kernel(uint64_t* counts, const size_t n,
                                            const uint64_t max_edges) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t       idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < n) {
    counts[idx] = std::min(counts[idx], max_edges);
    idx += stride;
  }
}

// Writes the final CSR sentinel offset using scanned counts.
SALTATLAS_HD_GLOBAL void set_csr_last_offset_kernel(uint64_t*       offsets,
                                                    const uint64_t* counts,
                                                    const size_t n_vertices) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (n_vertices == 0) {
      offsets[0] = 0;
    } else {
      offsets[n_vertices] = offsets[n_vertices - 1] + counts[n_vertices - 1];
    }
  }
}

// Casts 64-bit CSR offsets down to the output offset type.
template <typename id_type>
SALTATLAS_HD_GLOBAL void cast_offsets_kernel(const uint64_t* in_offsets,
                                             id_type*        out_offsets,
                                             const size_t    n) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t       idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < n) {
    out_offsets[idx] = static_cast<id_type>(in_offsets[idx]);
    idx += stride;
  }
}

// Zero-initializes per-vertex write cursors for edge filling.
SALTATLAS_HD_GLOBAL void zero_u32_kernel(uint32_t* data, const size_t n) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t       idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  while (idx < n) {
    data[idx] = 0;
    idx += stride;
  }
}

// Fills reverse CSR edges (source IDs and values) with bounded per-vertex
// writes.
template <typename id_type, typename dist_type>
SALTATLAS_HD_GLOBAL void fill_reverse_csr_kernel(
    const matrix_view<const id_type>   in_knng_ids,
    const matrix_view<const dist_type> in_knng_values, const uint64_t* offsets,
    uint32_t* write_counts, id_type* out_ids, dist_type* out_distances) {
  const size_t total_edges = in_knng_ids.size();
  const size_t degree      = in_knng_ids.n_cols();
  const size_t stride      = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t       idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  while (idx < total_edges) {
    const size_t  sid = idx / degree;
    const size_t  col = idx - sid * degree;
    const id_type nid = in_knng_ids(sid, col);
    if (is_valid_vertex_id(nid, in_knng_ids.n_rows())) {
      const size_t   nid_idx = static_cast<size_t>(nid);
      const uint64_t begin   = offsets[nid_idx];
      const uint64_t end     = offsets[nid_idx + 1];
      const uint64_t limit   = end - begin;
      const uint32_t pos     = atomicAdd(&write_counts[nid_idx], 1U);
      if (static_cast<uint64_t>(pos) < limit) {
        const size_t out_pos   = static_cast<size_t>(begin + pos);
        out_ids[out_pos]       = static_cast<id_type>(sid);
        out_distances[out_pos] = in_knng_values(sid, col);
      }
    }
    idx += stride;
  }
}

// Builds a 1D launch grid with an upper bound on block count.
inline dim3 make_grid_1d(const size_t n_items, const int block_size) {
  constexpr size_t k_max_blocks = 65535;
  if (n_items == 0) {
    return dim3(1);
  }
  const size_t n_blocks = (n_items + block_size - 1) / block_size;
  return dim3(static_cast<unsigned int>(std::min(k_max_blocks, n_blocks)));
}

}  // namespace detail

// Builds a bounded reverse graph on APU/ROCm from dense KNNG IDs/values.
template <typename id_type, typename dist_type>
inline csr_graph<id_type, dist_type> make_reversed_graph_apu(
    const matrix_view<id_type>&   in_knng_ids,
    const matrix_view<dist_type>& in_knng_values,
    const size_t                  max_edges_per_vertex) {
  static_assert(std::is_integral_v<id_type>, "id_type must be integral");

  if (in_knng_ids.n_rows() != in_knng_values.n_rows() ||
      in_knng_ids.n_cols() != in_knng_values.n_cols()) {
    throw std::invalid_argument(
        "in_knng_ids and in_knng_values shape mismatch");
  }

  const size_t                  n_vertices = in_knng_ids.n_rows();
  csr_graph<id_type, dist_type> out;
  out.offsets = make_hip_array<id_type>(n_vertices + 1);

  if (n_vertices == 0 || max_edges_per_vertex == 0) {
    SALTATLAS_HIP_CHECK(
        hipMemset(out.offsets.get(), 0, (n_vertices + 1) * sizeof(id_type)));
    return out;
  }

  auto counts = make_hip_array<uint64_t>(n_vertices);
  SALTATLAS_HIP_CHECK(
      hipMemset(counts.get(), 0, n_vertices * sizeof(uint64_t)));

  constexpr int k_block_size = 256;
  const auto    count_grid =
      detail::make_grid_1d(in_knng_ids.size(), k_block_size);
  hipLaunchKernelGGL(
      (detail::count_reverse_edges_kernel<id_type>), count_grid,
      dim3(k_block_size), 0, nullptr,
      matrix_view<const id_type>(in_knng_ids.data(), in_knng_ids.n_rows(),
                                 in_knng_ids.n_cols()),
      counts.get());
  SALTATLAS_HIP_CHECK(hipGetLastError());

  const auto clip_grid = detail::make_grid_1d(n_vertices, k_block_size);
  hipLaunchKernelGGL((detail::clip_counts_kernel), clip_grid,
                     dim3(k_block_size), 0, nullptr, counts.get(), n_vertices,
                     static_cast<uint64_t>(max_edges_per_vertex));
  SALTATLAS_HIP_CHECK(hipGetLastError());

  auto   offsets64    = make_hip_array<uint64_t>(n_vertices + 1);
  size_t scan_tmp_len = 0;
  SALTATLAS_HIP_CHECK(rocprim::exclusive_scan(nullptr, scan_tmp_len,
                                              counts.get(), offsets64.get(),
                                              uint64_t{0}, n_vertices));
  auto scan_tmp = make_hip_array<char>(scan_tmp_len);
  SALTATLAS_HIP_CHECK(rocprim::exclusive_scan(scan_tmp.get(), scan_tmp_len,
                                              counts.get(), offsets64.get(),
                                              uint64_t{0}, n_vertices));

  hipLaunchKernelGGL((detail::set_csr_last_offset_kernel), dim3(1), dim3(1), 0,
                     nullptr, offsets64.get(), counts.get(), n_vertices);
  SALTATLAS_HIP_CHECK(hipGetLastError());

  uint64_t total_edges_u64 = 0;
  SALTATLAS_HIP_CHECK(hipMemcpy(&total_edges_u64, offsets64.get() + n_vertices,
                                sizeof(uint64_t), hipMemcpyDeviceToHost));
  if (total_edges_u64 >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("total reversed edges exceed size_t range");
  }
  if (total_edges_u64 >
      static_cast<uint64_t>(std::numeric_limits<id_type>::max())) {
    throw std::overflow_error(
        "total reversed edges exceed id_type range; use wider id_type");
  }

  const auto cast_grid = detail::make_grid_1d(n_vertices + 1, k_block_size);
  hipLaunchKernelGGL((detail::cast_offsets_kernel<id_type>), cast_grid,
                     dim3(k_block_size), 0, nullptr, offsets64.get(),
                     out.offsets.get(), n_vertices + 1);
  SALTATLAS_HIP_CHECK(hipGetLastError());

  const size_t total_edges = static_cast<size_t>(total_edges_u64);
  out.ids                  = make_hip_array<id_type>(total_edges);
  out.distances            = make_hip_array<dist_type>(total_edges);

  if (total_edges == 0) {
    return out;
  }

  auto write_counts = make_hip_array<uint32_t>(n_vertices);
  hipLaunchKernelGGL((detail::zero_u32_kernel), clip_grid, dim3(k_block_size),
                     0, nullptr, write_counts.get(), n_vertices);
  SALTATLAS_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(
      (detail::fill_reverse_csr_kernel<id_type, dist_type>), count_grid,
      dim3(k_block_size), 0, nullptr,
      matrix_view<const id_type>(in_knng_ids.data(), in_knng_ids.n_rows(),
                                 in_knng_ids.n_cols()),
      matrix_view<const dist_type>(in_knng_values.data(),
                                   in_knng_values.n_rows(),
                                   in_knng_values.n_cols()),
      offsets64.get(), write_counts.get(), out.ids.get(), out.distances.get());
  SALTATLAS_HIP_CHECK(hipGetLastError());

  return out;
}

// Builds a reverse graph using the default cap (2x forward degree).
template <typename id_type, typename dist_type>
inline csr_graph<id_type, dist_type> make_reversed_graph_apu(
    const matrix_view<id_type>&   in_knng_ids,
    const matrix_view<dist_type>& in_knng_values) {
  const size_t max_edges_per_vertex = in_knng_ids.n_cols() * 2;
  return make_reversed_graph_apu<id_type, dist_type>(
      in_knng_ids, in_knng_values, max_edges_per_vertex);
}

}  // namespace saltatlas::solanet::apu_nn