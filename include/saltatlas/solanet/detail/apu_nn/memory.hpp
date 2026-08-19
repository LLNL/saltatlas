// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#if !defined(__CUDACC__)
#include <hip/hip_runtime.h>
#endif
#include <spdlog/spdlog.h>

#include "saltatlas/solanet/detail/apu_nn/utils.hpp"

namespace saltatlas::solanet::apu_nn {
// Basic utility functions for HIP/ROCM backend
template <typename T>
T* allocate_hip_memory(const size_t n_elements) {
  // Allocate memory on HIP device
  T* d_ptr = nullptr;

  spdlog::trace("Allocating HIP memory: {} GB", n_elements * sizeof(T) / 1e9);
  SALTATLAS_HIP_CHECK(hipMalloc(&d_ptr, n_elements * sizeof(T)));
  spdlog::trace("HIP memory allocated at address: {}", (void*)d_ptr);
  return d_ptr;
}

// template <typename T>
// T* allocate_hip_managed_memory(const size_t n_elements) {
//   // Allocate unified memory accessible from host and device.
//   T* d_ptr = nullptr;
//   SALTATLAS_HIP_CHECK(
//       hipMallocManaged(&d_ptr, n_elements * sizeof(T), hipMemAttachGlobal));
//   return d_ptr;
// }

template <typename T>
void free_hip_memory(T* d_ptr) {
  // Free memory on HIP device
  if (d_ptr != nullptr) {
    spdlog::trace("Freeing HIP memory at address: {}", (void*)d_ptr);
    SALTATLAS_HIP_CHECK(hipFree(d_ptr));
  }
}

template <typename T>
struct hip_deleter {
  void operator()(T* d_ptr) const noexcept { free_hip_memory(d_ptr); }
};

template <typename T>
using hip_unique_ptr = std::unique_ptr<T, hip_deleter<T>>;

template <typename T>
hip_unique_ptr<T> make_hip_array(const size_t n_elements) {
  return hip_unique_ptr<T>(allocate_hip_memory<T>(n_elements));
}

// STL-compatible allocator that uses hipMalloc internally
template <typename T>
class hip_allocator {
 public:
  using value_type = T;

  hip_allocator() = default;

  ~hip_allocator() = default;

  template <typename U>
  hip_allocator(const hip_allocator<U>&) noexcept {}

  T* allocate(const size_t n) { return allocate_hip_memory<T>(n); }

  void deallocate(T* p, size_t /*n*/) noexcept { free_hip_memory(p); }
};

// Allocate aligned array from shared memory
// Advances the pointer accordingly
// Returns the allocated pointer
// Assumes that there is enough space in shared memory.
// No bounds checking is performed.
template <typename T>
SALTATLAS_HD_DEVICE SALTATLAS_HD_FORCEINLINE T* get_shared_mem(
    unsigned char*& ptr, const int count, const int n_warps_per_block,
    const int local_warp_id) {
  const uintptr_t addr         = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t aligned_addr = (addr + alignof(T) - 1) & ~(alignof(T) - 1);
  ptr                          = reinterpret_cast<unsigned char*>(aligned_addr);
  T* const typed_ptr = reinterpret_cast<T*>(ptr) + local_warp_id * count;
  ptr += sizeof(T) * count * n_warps_per_block;
  return typed_ptr;
};

// TODO: Stop using this version.
// Allocate aligned array from shared memory
// Advances the pointer accordingly
// Returns the allocated pointer
// Assumes that there is enough space in shared memory.
// No bounds checking is performed.
// template <typename T>
// SALTATLAS_HD_DEVICE inline T* get_shared_mem(unsigned char*& ptr,
//                                              const int       count) {
//   const uintptr_t addr         = reinterpret_cast<uintptr_t>(ptr);
//   const uintptr_t aligned_addr = (addr + alignof(T) - 1) & ~(alignof(T) - 1);
//   ptr                          = reinterpret_cast<unsigned
//   char*>(aligned_addr); T* const typed_ptr           =
//   reinterpret_cast<T*>(ptr); ptr += sizeof(T) * count; return typed_ptr;
// }

}  // namespace saltatlas::solanet::apu_nn
