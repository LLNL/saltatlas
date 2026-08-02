// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/solanet/detail/cuvs_nn/common.hpp>
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
  bool                  l2_normalize{false};
  size_t                k{0};
  double rho{0.5};  // Not used in cuVS, but kept for consistency.
  double delta{0.0001};
  int    max_iterations{100};
  double rmm_pool_size_gb{-1};
  // Distance-compute dtype: auto | fp32 | fp16.
  // cuVS defaults to AUTO, which selects the fp16 tensor-core kernel
  // (local_join_kernel_wmma) whenever dim > 16. fp32 forces the scalar
  // kernel (local_join_kernel_simt), which is what SOLANET uses.
  std::string dist_dtype{"auto"};
  bool        optimize{false};
  std::filesystem::path output_path{};
  bool                  dump_distance{false};
  bool                  verbose{false};

  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "l2_normalize: " << l2_normalize << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "rho: " << rho << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "max_iterations: " << max_iterations << std::endl;
    std::cout << "rmm_pool_size_gb: " << rmm_pool_size_gb << std::endl;
    std::cout << "dist_dtype: " << dist_dtype << std::endl;
    std::cout << "optimize: " << optimize << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "dump_distance: " << dump_distance << std::endl;
    std::cout << "verbose: " << verbose << std::endl;
  }
};

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:d:r:m:M:T:oG:N:Dvh")) != -1) {
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
      case 'N':
        opt.l2_normalize = true;
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
      case 'T':
        opt.dist_dtype = optarg;
        break;
      case 'o':
        opt.optimize = true;
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
               "product), default: l2."
            << std::endl;
  std::cout
      << "  -N: L2 normalize points before building the index (default: false)"
      << std::endl;
  std::cout << "  -k <k>: Number of neighbors (required)" << std::endl;
  std::cout
      << "  -d <delta>: Termination threshold for nn-descent (default: 0.0001)"
      << std::endl;
  std::cout
      << "  -r <rho>: Sampling rate for candidate neighbors in nn-descent "
         "(default: 0.5, not used in cuVS)"
      << std::endl;
  std::cout
      << "  -m <max_iterations>: Maximum iterations for nn-descent (default: "
         "100)"
      << std::endl;
  std::cout
      << "  -M <rmm_pool_size_gb>: RMM pool size in GB (default: use all free "
         "memory with some margin)"
      << std::endl;
  std::cout
      << "  -T <auto|fp32|fp16>: dtype for distance computation (default: "
         "auto, which uses fp16 tensor cores when dim > 16)"
      << std::endl;
  std::cout
      << "  -o: Optimize the kNNG by pruning high-degree points and keeping "
         "only the closest neighbors (default: false)"
      << std::endl;
  std::cout << "  -G <output_path>: Directory path to dump the built kNNG."
               "(default: empty, no dumping)"
            << std::endl;
  std::cout
      << "  -D: Dump distances along with neighbor IDs when dumping kNNG with "
         "-G option (default: false)"
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
  // RMM's pool constructor that takes a device_async_resource_ref is a
  // template whose Upstream parameter cannot be deduced from the ref, so name
  // it explicitly. (Class template argument deduction works only with the
  // Upstream* overload.)
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> rmm_pool(
      rmm::mr::get_current_device_resource_ref(),
      static_cast<std::size_t>(pool_size));
  rmm::mr::set_current_device_resource(rmm_pool);

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

  if (opt.l2_normalize) {
    l2_normalize_points(points.data(), points.num_points(),
                        points.num_dimensions());
  }

  using namespace saltatlas::solanet::cuvs_nn;
  raft::resources        host_res;
  raft::device_resources dev_res;

  const size_t n_points = points.num_points();
  const size_t n_dims   = points.num_dimensions();

  // cuVS CUDA path requires explicit host-to-device copy before building.
  auto h_dataset = raft::make_host_matrix<fe_type, int64_t>(n_points, n_dims);
  raft::copy(h_dataset.data_handle(), points.data(), n_points * n_dims,
             raft::resource::get_cuda_stream(dev_res));
  raft::resource::sync_stream(dev_res,
                              raft::resource::get_cuda_stream(dev_res));

  cuvs::distance::DistanceType dist_func =
      (opt.distance_function == "l2")
          ? cuvs::distance::DistanceType::L2Expanded
          : ((opt.distance_function == "ip")
                 ? cuvs::distance::DistanceType::InnerProduct
                 : cuvs::distance::DistanceType::L2Expanded);

  std::cout << "\nBuild KNNG" << std::endl;
  saltatlas::rec_time().start("Build-knng");

  auto nnd_params =
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params(opt.k,
                                                                    dist_func);
  nnd_params.graph_degree              = opt.k;
  nnd_params.intermediate_graph_degree = opt.k;
  nnd_params.return_distances          = true;
  nnd_params.max_iterations            = opt.max_iterations;
  nnd_params.termination_threshold     = opt.delta;

  // Distance dtype selection. cuVS gained `dist_comp_dtype` after v25.10, so
  // detect it at compile time and degrade gracefully on older installs
  // (25.10 has only the fp16 WMMA local-join kernel; there is no fp32 path).
  if constexpr (requires { nnd_params.dist_comp_dtype; }) {
    using DCT = std::decay_t<decltype(nnd_params.dist_comp_dtype)>;
    if (opt.dist_dtype == "fp32") {
      nnd_params.dist_comp_dtype = DCT::FP32;  // -> local_join_kernel_simt
    } else if (opt.dist_dtype == "fp16") {
      nnd_params.dist_comp_dtype = DCT::FP16;  // -> local_join_kernel_wmma
    }  // "auto" keeps the cuVS default
  } else if (opt.dist_dtype != "auto") {
    std::cerr << "WARNING: installed cuVS has no dist_comp_dtype; -T ignored. "
                 "Distances are computed in fp16 on tensor cores."
              << std::endl;
  }

  saltatlas::rec_time().start("Copy-pstore-to-dev");
  auto d_pstore = copy_to_dev(make_host_matrix_view(h_dataset), dev_res);
  saltatlas::rec_time().stop();

  saltatlas::rec_time().start("nnd-kernel");
  auto index = cuvs::neighbors::nn_descent::build(
      dev_res, nnd_params, make_const_matrix_view(d_pstore));
  saltatlas::rec_time().stop();

  if (!index.distances().has_value()) {
    std::cerr << "nn_descent index does not contain distances." << std::endl;
    return EXIT_FAILURE;
  }

  saltatlas::rec_time().start("Copy-knng-to-host");
  auto h_nids  = copy_to_host(index.graph(), host_res, dev_res);
  auto h_dists = copy_to_host(*index.distances(), host_res, dev_res);
  saltatlas::rec_time().stop();
  saltatlas::rec_time().stop();  // build_knng

  auto h_nids_view  = make_host_matrix_view(h_nids);
  auto h_dists_view = make_host_matrix_view(h_dists);

  // show_index_score(h_dists.data_handle(), n_points, opt.k, false);

  print_time_table();
  saltatlas::rec_time().reset();

  if (!opt.output_path.empty()) {
    std::cout << "\nDump KNNG to " << opt.output_path << std::endl;
    const auto ids_view =
        host_matrix_view<id_type>(h_nids_view.data_handle(), n_points, opt.k);
    const auto dists_view = host_matrix_view<dist_type>(
        h_dists_view.data_handle(), n_points, opt.k);
    dump_knng(ids_view, dists_view, opt.output_path, opt.dump_distance);
  }

  return 0;
}
