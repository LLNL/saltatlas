// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// SOLANET (lock-free NN-Descent) on a single NVIDIA GPU.
// The kernels are the same apu_nn sources used on MI300A; they compile for
// CUDA through the backend layer in apu_nn/utils.hpp. Unlike the APU driver,
// points are staged in pinned host memory and copied to the device
// explicitly, and the built KNNG is copied back before scoring/dumping.

#define SALTATLAS_SOLANET_APU_NND_TEAM_SIZE 8

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// RMM pool setup mirrors run_cuvs_single_gpu_nndescent.cpp. RMM moved these
// headers (rmm/mr/device/... -> rmm/mr/...) and switched to the *_ref resource
// API, so enable the pool only when the newer layout is present; older stacks
// (e.g. the RMM bundled with cuVS 25.10) simply build without it.
// Note: SOLANET allocates its handful of large buffers directly through
// apu_nn/memory.hpp (cudaMalloc), so the pool is for parity and future use,
// not a performance factor for this driver.
#if __has_include(<rmm/mr/pool_memory_resource.hpp>) && \
    __has_include(<rmm/mr/per_device_resource.hpp>)
#define SALTATLAS_SOLANET_USE_RMM 1
#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#endif

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/solanet/detail/apu_nn/matrix.hpp>
#include <saltatlas/solanet/detail/apu_nn/memory.hpp>
#include <saltatlas/solanet/detail/apu_nn/nndescent.hpp>
#include <saltatlas/solanet/singleton_time_recorder.hpp>

#include "l2_normalize_points.hpp"
#include "solanet_shm_common.hpp"

template <typename T>
class host_matrix_view {
 public:
  host_matrix_view(T* const data, const size_t n_rows, const size_t n_cols)
      : m_data(data), m_n_rows(n_rows), m_n_cols(n_cols) {}

  T& operator()(const size_t row, const size_t col) {
    return m_data[row * m_n_cols + col];
  }

  const T& operator()(const size_t row, const size_t col) const {
    return m_data[row * m_n_cols + col];
  }

  size_t n_rows() const { return m_n_rows; }
  size_t n_cols() const { return m_n_cols; }

 private:
  T*     m_data   = nullptr;
  size_t m_n_rows = 0;
  size_t m_n_cols = 0;
};

template <typename T>
class cuda_pinned_allocator {
 public:
  using value_type        = T;
  cuda_pinned_allocator() = default;
  template <typename U>
  cuda_pinned_allocator(const cuda_pinned_allocator<U>&) noexcept {}

  T* allocate(const size_t n) {
    void*      ptr = nullptr;
    const auto err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t /*n*/) noexcept {
    if (p) {
      cudaFreeHost(p);
    }
  }
};

struct options {
  std::filesystem::path point_files_path;
  std::string           point_file_format;
  std::string           distance_function{"l2"};
  size_t                k{0};
  double                rho{0.5};
  double                delta{0.0001};
  int                   max_iterations{100};
  double                rmm_pool_size_gb{-1};
  std::filesystem::path output_path{};
  bool                  dump_distance{false};
  bool                  verbose{false};

  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "rho: " << rho << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "max_iterations: " << max_iterations << std::endl;
    std::cout << "rmm_pool_size_gb: " << rmm_pool_size_gb << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "dump_distance: " << dump_distance << std::endl;
    std::cout << "verbose: " << verbose << std::endl;
  }
};

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:d:r:m:M:G:Dvh")) != -1) {
    switch (p) {
      case 'i':
        opt.point_files_path = std::filesystem::path(optarg);
        break;
      case 'p':
        opt.point_file_format = optarg;
        break;
      case 'f':
        opt.distance_function = optarg;
        break;
      case 'k':
        opt.k = std::stoul(optarg);
        break;
      case 'd':
        opt.delta = std::stod(optarg);
        break;
      case 'r':
        opt.rho = std::stod(optarg);
        break;
      case 'm':
        opt.max_iterations = std::stoi(optarg);
        break;
      case 'M':
        opt.rmm_pool_size_gb = std::stod(optarg);
        break;
      case 'G':
        opt.output_path = optarg;
        break;
      case 'D':
        opt.dump_distance = true;
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'h':
        show_usage = true;
        return true;
      default:
        show_usage = true;
        return false;
    }
  }

  if (opt.point_files_path.empty() || opt.point_file_format.empty() ||
      opt.distance_function.empty() || opt.k == 0) {
    std::cerr << "Error: Missing required options." << std::endl;
    return false;
  }
  if (opt.distance_function != "l2" && opt.distance_function != "ip") {
    std::cerr << "Error: Unsupported distance function: "
              << opt.distance_function << std::endl;
    return false;
  }
  if (opt.max_iterations <= 0) {
    std::cerr << "Error: max_iterations must be > 0." << std::endl;
    return false;
  }
  return true;
}

void show_usage(const char* prog_name) {
  std::cout
      << "Usage: " << prog_name
      << " -i <point_files_path> -p <point_file_format> -f <distance_function> "
         "-k <k> [options]"
      << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -i <point_files_path>: Path to point files (required)"
            << std::endl;
  std::cout << "  -p <point_file_format>: Format of point files (required)"
            << std::endl;
  std::cout << "  -f <distance_function>: Distance function. l2 or ip (inner "
               "product), default: l2. Points are L2 normalized "
               "automatically for ip."
            << std::endl;
  std::cout << "  -k <k>: Number of neighbors (required)" << std::endl;
  std::cout
      << "  -d <delta>: Termination threshold for nn-descent (default: 0.0001)"
      << std::endl;
  std::cout << "  -r <rho>: Sampling rate for candidate neighbors in "
               "nn-descent (default: 0.5)"
            << std::endl;
  std::cout
      << "  -m <max_iterations>: Maximum iterations for nn-descent (default: "
         "100)"
      << std::endl;
  std::cout
      << "  -M <rmm_pool_size_gb>: RMM pool size in GB (default: use all free "
         "memory with some margin)"
      << std::endl;
  std::cout << "  -G <output_path>: If specified, dump the KNNG to the given "
               "path. Distance will be dumped if -D is also specified."
            << std::endl;
  std::cout << "  -D: Dump distances along with neighbor IDs when dumping "
               "KNNG. Only effective if -G is also specified."
            << std::endl;
  std::cout << "  -v: Verbose output (default: false)" << std::endl;
  std::cout << "  -h: Show this help message and exit" << std::endl;
}

int main(int argc, char* argv[]) {
  options opt;
  bool    show_help = false;
  if (!parse_options(argc, argv, opt, show_help)) {
    show_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (show_help) {
    show_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  opt.show();
  std::cout << std::endl;

  int num_gpus = 0;
  if (cudaGetDeviceCount(&num_gpus) != cudaSuccess || num_gpus == 0) {
    std::cerr << "No CUDA devices found." << std::endl;
    return EXIT_FAILURE;
  }
  const int dev = 0;
  if (cudaSetDevice(dev) != cudaSuccess) {
    std::cerr << "Failed to set CUDA device " << dev << std::endl;
    return EXIT_FAILURE;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    std::cerr << "Failed to get CUDA device properties." << std::endl;
    return EXIT_FAILURE;
  }
  if (opt.verbose) {
    std::cout << "Using device " << dev << "/" << num_gpus << ": " << prop.name
              << " (" << static_cast<double>(prop.totalGlobalMem) / (1ULL << 30)
              << " GB)" << std::endl;
  }

#ifdef SALTATLAS_SOLANET_USE_RMM
  double pool_size = 0.0;
  if (opt.rmm_pool_size_gb > 0) {
    pool_size = opt.rmm_pool_size_gb * (1ULL << 30);
  } else {
    pool_size = rmm::percent_of_free_device_memory(80);
  }
  if (opt.verbose) {
    std::cout << "RMM pool size (GB): "
              << pool_size / static_cast<double>(1ULL << 30) << std::endl;
  }
  rmm::mr::pool_memory_resource rmm_pool(
      rmm::mr::get_current_device_resource_ref(),
      static_cast<std::size_t>(pool_size));
  rmm::mr::set_current_device_resource(rmm_pool);
#else
  if (opt.verbose) {
    std::cout << "RMM pool: disabled (installed RMM predates the current "
                 "header/resource API; SOLANET allocates via cudaMalloc)"
              << std::endl;
  }
#endif

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  auto points =
      saltatlas::load_points<id_type, id_type, fe_type, e2i_id_map_type,
                             cuda_pinned_allocator<fe_type>>(
          point_file_paths, opt.point_file_format);
  std::cout << "Number of points: " << points.num_points() << std::endl;
  std::cout << "Number of dimensions: " << points.num_dimensions() << std::endl;
  if (points.num_points() == 0 || points.num_dimensions() == 0) {
    std::cerr << "No points loaded." << std::endl;
    return EXIT_FAILURE;
  }

  if (opt.distance_function == "ip") {
    std::cout << "!!! L2 normalizing points for inner product search... !!"
              << std::endl;
    l2_normalize_points(points.data(), points.num_points(),
                        points.num_dimensions());
  }

  using namespace saltatlas::solanet;

  const size_t n_points = points.num_points();
  const size_t n_dims   = points.num_dimensions();

  std::cout << "\nBuild KNNG" << std::endl;
  saltatlas::rec_time().start("Build-knng");

  // Discrete GPU: copy the point store to device memory explicitly.
  saltatlas::rec_time().start("Copy-pstore-to-dev");
  auto d_pstore = apu_nn::make_hip_array<fe_type>(n_points * n_dims);
  SALTATLAS_HIP_CHECK(cudaMemcpy(d_pstore.get(), points.data(),
                                 n_points * n_dims * sizeof(fe_type),
                                 cudaMemcpyHostToDevice));
  saltatlas::rec_time().stop();

  apu_nn::matrix_view<fe_type> pstore_view(d_pstore.get(), n_points, n_dims);
  auto [knn_ids, knn_dists] =
      apu_nn::build_index<id_type, fe_type, dist_type>(
          pstore_view, opt.distance_function, opt.k, opt.rho, opt.delta,
          std::random_device{}(), opt.max_iterations);

  // Copy the built KNNG back to (pinned) host memory.
  saltatlas::rec_time().start("Copy-knng-to-host");
  std::vector<id_type, cuda_pinned_allocator<id_type>> h_ids(n_points * opt.k);
  std::vector<dist_type, cuda_pinned_allocator<dist_type>> h_dists(n_points *
                                                                   opt.k);
  SALTATLAS_HIP_CHECK(cudaMemcpy(h_ids.data(), knn_ids.data(),
                                 h_ids.size() * sizeof(id_type),
                                 cudaMemcpyDeviceToHost));
  SALTATLAS_HIP_CHECK(cudaMemcpy(h_dists.data(), knn_dists.data(),
                                 h_dists.size() * sizeof(dist_type),
                                 cudaMemcpyDeviceToHost));
  saltatlas::rec_time().stop();
  saltatlas::rec_time().stop();  // Build-knng

  // show_index_score(h_dists.data(), n_points, opt.k, false);

  print_time_table();
  saltatlas::rec_time().reset();

  if (!opt.output_path.empty()) {
    std::cout << "\nDump KNNG to " << opt.output_path << std::endl;
    const auto ids_view =
        host_matrix_view<id_type>(h_ids.data(), n_points, opt.k);
    const auto dists_view =
        host_matrix_view<dist_type>(h_dists.data(), n_points, opt.k);
    dump_knng(ids_view, dists_view, opt.output_path, opt.dump_distance);
  }

  return 0;
}
