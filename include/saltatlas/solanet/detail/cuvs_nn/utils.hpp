// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>
#include <unistd.h>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <raft/core/device_resources.hpp>

#define SOLANET_CUDA_CHECK(call)                                    \
  {                                                                 \
    const cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                                     \
      std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                << cudaGetErrorString(error) << std::endl;          \
      exit(1);                                                      \
    }                                                               \
  }

namespace saltatlas::solanet::gpu {

/// \brief Update kNN graph on GP. All arrays are assumed to be allocated on GPU
/// and sorted. The function will update the master neighbor ids and distances
/// with new candidate neighbors.
/// \param master_nids Array of master neighbor ids.
/// \param master_dists Array of master distances.
/// \param candidate_nids Array of new candiate neighbors' ids .
/// \param candidate_dists Array of distances of candiate neighbors.
/// \param n_points Number of points.
/// \param degree Degree of the graph.
/// \return Number of neighbor updates.
template <typename id_type, typename dist_type>
size_t update_knng(id_type* master_nids, dist_type* master_dists,
                   const id_type*   candidate_nids,
                   const dist_type* candidate_dists, size_t n_points,
                   int degree, raft::device_resources& dev_resources);

/// \brief Add a scalar to an array.
/// \param data Array to add a scalar, allocated on GPU.
/// \param scalar Scalar to add.
/// \param length Length of the array.
template <typename T>
void add_scalar(T* data, const T scalar, const size_t length);

/// \brief Copy a 2D array from host to device.
/// \tparam T Data type.
/// \param d_dst Destination array, allocated on GPU.
/// \param dst_n_cols Number of columns of the destination array.
/// \param h_src Source array, allocated on CPU.
/// \param src_n_cols Number of columns of the source array.
/// \param n_rows Number of rows to copy.
template <typename T>
void copy_to_device_2d(T* const d_dst, const size_t dst_n_cols,
                       const T* const h_src, const size_t src_n_cols,
                       const size_t n_rows) {
  const size_t dst_pitch      = dst_n_cols * sizeof(T);
  const size_t src_pitch      = src_n_cols * sizeof(T);
  const size_t width_in_bytes = dst_n_cols * sizeof(T);
  const size_t height         = n_rows;

  SOLANET_CUDA_CHECK(cudaMemcpy2D(d_dst, dst_pitch, h_src, src_pitch,
                                  width_in_bytes, height,
                                  cudaMemcpyHostToDevice));
}

inline std::string get_cuvs_gpu_info(const int node_local_rank) {
  // Get number of GPUs
  int num_gpus;
  SOLANET_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

  // Assign GPU to MPI rank (round-robin)
  int gpu_id = node_local_rank % num_gpus;
  SOLANET_CUDA_CHECK(cudaSetDevice(gpu_id));

  // Get GPU UUID
  cudaDeviceProp prop;
  SOLANET_CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

  std::stringstream ss_uuid;
  for (int i = 0; i < 16; ++i) {
    ss_uuid << std::hex << static_cast<int>(prop.uuid.bytes[i]) << " ";
  }

  std::stringstream ss;
  ss << " GPU " << gpu_id << " /  " << num_gpus << ", " << prop.name << ", "
     << (float)prop.totalGlobalMem / (1ULL << 30) << " GB, "
     << " UUID " << ss_uuid.str();

  return ss.str();
}

}  // namespace saltatlas::solanet::gpu