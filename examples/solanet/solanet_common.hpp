// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

// Include only when available
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#endif
#if __has_include(<ygm/comm.hpp>)
#include <ygm/comm.hpp>
#endif

#ifdef SALTATLAS_SOLANET_CUVS
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#endif

#include <metall/detail/memory.hpp>

#include "saltatlas/dnnd/detail/utilities/file.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/dnnd/distance.hpp"
#include "saltatlas/neo_dnnd/mpi.hpp"
#include "saltatlas/solanet/singleton_time_recorder.hpp"
#include "saltatlas/solanet/solanet.hpp"

#ifndef NDEBUG
#include <saltatlas/common/detail/utilities/backtrace.hpp>
#endif

using namespace saltatlas;

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using dist_type = float;

#ifdef SALTATLAS_SOLANET_CUVS
using solanet_engine =
    solanet::solanet_engine_cuvs<id_type, fe_type, dist_type>;
#elif defined(SALTATLAS_SOLANET_OMP)
using solanet_engine = solanet::solanet_engine_omp<id_type, fe_type, dist_type>;
#elif defined(SALTATLAS_SOLANET_APU)
using solanet_engine = solanet::solanet_engine_apu<id_type, fe_type, dist_type>;
#else
#error \
    "SALTATLAS_SOLANET_CUVS, SALTATLAS_SOLANET_OMP, or SALTATLAS_SOLANET_APU must be defined."
#endif

using knng_type      = typename solanet_engine::sparse_knng_type;
using neighbor_type  = typename solanet_engine::neighbor_type;
using dist_func_type = distance::distance_function_type<fe_type, dist_type>;

#if defined(SALTATLAS_SOLANET_CUVS) || defined(SALTATLAS_SOLANET_APU)
using rmm_mem_pool_t =
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
#endif

inline void setup_spdlog(std::ostream& os, const bool verbose = false) {
  auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(os);

  auto logger = std::make_shared<spdlog::logger>("SOLANET", ostream_sink);

  spdlog::set_default_logger(logger);

  spdlog::flush_on(spdlog::level::warn);
  if (verbose) {
    spdlog::set_level(spdlog::level::trace);
    return;
  }
#ifndef NDEBUG
  spdlog::set_level(spdlog::level::debug);
#else
  spdlog::set_level(spdlog::level::info);
#endif
}

inline void show_common_config(mpi::communicator& comm) {
#ifdef NDEBUG
  comm.cout0() << "Build mode: Release" << std::endl;
#else
  comm.cout0() << "Build mode: Debug" << std::endl;
#endif
}

#if defined(SALTATLAS_SOLANET_CUVS) || defined(SALTATLAS_SOLANET_APU)
inline void show_gpu_config(mpi::communicator& comm) {
  comm.cout0() << saltatlas::solanet::gpu::get_cuvs_gpu_info(
                      comm.node_local_rank())
               << std::endl;
}

// If dev_pool_size_gb <= 0, set the pool size automatically
inline std::unique_ptr<rmm_mem_pool_t> setup_rmm_pool(
    const double dev_pool_size_gb, mpi::communicator& comm) {
  const auto [avail, total] = rmm::available_device_memory();
  comm.cout0() << "Available device memory (GB): " << avail / (double)(1 << 30)
               << std::endl;
  comm.cout0() << "Total device memory (GB): " << total / (double)(1 << 30)
               << std::endl;

  // Set up device memory pool for cuvs
  // 4 -> 4 GPUs per node
  comm.cout0() << "Setting up rmm pool, assuming 4 GPUs per node " << std::endl;
  const auto n_ranks_per_node = comm.node_size();
  int        n_ranks_per_device =
      (n_ranks_per_node <= 4) ? 1 : (n_ranks_per_node + 4 - 1) / 4;
  double dev_pool_size = 0;
  if (dev_pool_size_gb > 0) {
    dev_pool_size = dev_pool_size_gb * (1ULL << 30);
  } else {
    // Use 80% of GPU mem
    dev_pool_size = rmm::percent_of_free_device_memory(80 / n_ranks_per_device);
  }
  comm.cout0() << "Per rank dev pool size (GB): "
               << (double)dev_pool_size / (double)(1 << 30) << std::endl;
  comm.barrier();
  auto pool = std::make_unique<rmm_mem_pool_t>(
      rmm::mr::get_current_device_resource(), dev_pool_size);
  rmm::mr::set_current_device_resource(pool.get());

  // std::string                       filename{"logs-solanet-cuda.csv"};
  // rmm::mr::logging_resource_adaptor log_mr{
  //     rmm::mr::get_current_device_resource(), filename, true};

  return pool;
}
#endif

#ifdef SALTATLAS_SOLANET_APU
void setup_apu(mpi::communicator& comm, const bool verbose = false) {
  int device_count = 0;
  SALTATLAS_HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    comm.cerr() << "No APU device found." << std::endl;
    comm.abort();
  }

  // int apu_id = comm.node_local_rank() % device_count;
  int apu_id = 0;
  SALTATLAS_HIP_CHECK(hipSetDevice(apu_id));

  hipDeviceProp_t prop;
  SALTATLAS_HIP_CHECK(hipGetDeviceProperties(&prop, apu_id));

  std::string uuid_str;
  for (int i = 0; i < 16; ++i) {
    uuid_str += std::to_string(static_cast<int>(prop.uuid.bytes[i])) + " ";
  }

  if (verbose) {
    comm.cout() << " APU " << apu_id << " / " << device_count << ", "
                << uuid_str << ", " << (float)prop.totalGlobalMem / (1ULL << 30)
                << " GB" << std::endl;
  }

  // // Make sure there is only one APU per rank
  // if (device_count != 1) {
  //   comm.cerr()
  //       << "Multiple APUs detected. Please ensure there is only one APU "
  //          "per rank."
  //       << std::endl;
  //   comm.abort();
  // }
}
#endif

inline void show_omp_config(mpi::communicator& comm) {
  const int max_threads = std::thread::hardware_concurrency();
  if (max_threads == 0) {
    comm.cerr0() << "Failed to get the number of hardware threads."
                 << std::endl;
    comm.abort();
  }
  if (comm.rank() == 0) {
    OMP_DIRECTIVE(parallel) {
      const auto num_threads = solanet::omp::get_num_threads();
      OMP_DIRECTIVE(single) {
        std::cout << "#of max threads per node: " << max_threads << std::endl;
        std::cout << "#of threads per rank: " << num_threads << std::endl;
      }
    }
  }
}

// Set #of OpenMP threads per rank.
inline void set_omp_num_threads(mpi::communicator& comm,
                                const bool         verbose = false) {
  const int max_threads = std::thread::hardware_concurrency();
  if (max_threads == 0) {
    comm.cerr0() << "Failed to get the number of hardware threads."
                 << std::endl;
    comm.abort();
  }
  // Memo: just divide max_threads by comm.node_size() may not work on Tuo.
  // num_threads_per_rank = 42 could be the max value.
  const int num_threads_per_rank = max_threads / comm.node_size();
  comm.barrier();

  solanet::omp::set_num_threads(num_threads_per_rank);
  comm.barrier();
  if (verbose) {
    show_omp_config(comm);
  }
}

void show_dram_usage(mpi::communicator& comm, const size_t line_no) {
  comm.barrier();
  comm.cout0() << "DRAM usages at line " << line_no << std::endl;
  comm.cout0() << "DRAM used (GiB): "
               << metall::mtlldetail::get_used_ram_size() / double(1 << 30)
               << std::endl;
  comm.cout0() << "DRAM free (GiB): "
               << metall::mtlldetail::get_free_ram_size() / double(1 << 30)
               << std::endl;
  comm.cout0() << "DRAM cache (GiB): "
               << metall::mtlldetail::get_page_cache_size() / double(1 << 30)
               << std::endl;
  comm.barrier();
}

inline void show_index_score(const solanet_engine::sparse_knng_type& knng,
                             mpi::communicator&                      comm) {
  // Compute average neighbor distance
  double l_total_dist  = 0.0;
  size_t l_total_edges = 0;
  for (const auto& neighbors : knng) {
    for (const auto& neighbor : neighbors.second) {
      l_total_dist += static_cast<double>(neighbor.distance);
      l_total_edges++;
    }
  }

  const auto   g_total_dist  = comm.all_reduce_sum(l_total_dist);
  const auto   g_total_edges = comm.all_reduce_sum(l_total_edges);
  const double avg_dist = g_total_dist / static_cast<double>(g_total_edges);
  comm.cout0() << "Index average neighbor distance: " << avg_dist << std::endl;
}
