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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <cuvs/neighbors/cagra.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <saltatlas/dnnd/detail/utilities/file.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/solanet/detail/apu_nn/matrix.hpp>
#include <saltatlas/solanet/detail/cuvs_nn/common.hpp>
#include <saltatlas/solanet/singleton_time_recorder.hpp>

#include "l2_normalize_points.hpp"
#include "solanet_shm_common.hpp"

struct options {
  std::filesystem::path point_files_path;
  std::string           point_file_format;
  std::string           distance_function{"l2"};
  bool                  l2_normalize{false};
  size_t                k{0};
  double rho{0.5};  // Not used in cuVS, but we keep it for consistency with
                    // other implementations.
  double                delta{0.0001};
  int                   max_iterations{100};
  double                rmm_pool_size_gb{-1};
  bool                  optimize{false};
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
    std::cout << "optimize: " << optimize << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "dump_distance: " << dump_distance << std::endl;
    std::cout << "verbose: " << verbose << std::endl;
  }
};

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:d:r:m:M:oG:N:Dvh")) != -1) {
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
      << "  -o: Optimize the kNNG by pruning high-degree points and keeping "
         "only the closest neighbors (default: false)"
      << std::endl;
  std::cout
      << "  -G <output_path>: Directory path to dump the built kNNG. Each rank "
         "will dump its own part of the kNNG to a file in this directory."
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
    // show_usage(argv);
    return EXIT_FAILURE;
  }
  if (show_help) {
    show_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  opt.show();
  std::cout << std::endl;

  // Print device info
  int dev = 0;
  SALTATLAS_HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop{};
  SALTATLAS_HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  if (opt.verbose) {
    std::printf("Using device %d: %s (gfx: %s)\n", dev, prop.name,
                prop.gcnArchName);
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
  using rmm_mem_pool_t =
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  auto rmm_pool = std::make_unique<rmm_mem_pool_t>(
      rmm::mr::get_current_device_resource(), pool_size);
  rmm::mr::set_current_device_resource(rmm_pool.get());

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  auto points = saltatlas::load_points<
      id_type, id_type, fe_type, e2i_id_map_type,
      saltatlas::solanet::apu_nn::hip_allocator<fe_type>>(
      point_file_paths, opt.point_file_format);
  std::cout << "Number of points: " << points.num_points() << std::endl;
  std::cout << "Number of dimensions: " << points.num_dimensions() << std::endl;

  if (opt.l2_normalize) {
    l2_normalize_points(points.data(), points.num_points(),
                        points.num_dimensions());
  }

  using namespace saltatlas::solanet::cuvs_nn;
  raft::resources        host_res;
  raft::device_resources dev_res;

  const size_t n_points = points.num_points();
  const size_t n_dims   = points.num_dimensions();

  std::cout << "\nBuild KNNG" << std::endl;

  cuvs::distance::DistanceType dist_func =
      (opt.distance_function == "l2")
          ? cuvs::distance::DistanceType::L2Expanded
          : ((opt.distance_function == "ip")
                 ? cuvs::distance::DistanceType::InnerProduct
                 : cuvs::distance::DistanceType::L2Expanded);
  // CAGRA index parameters:
  // https://github.com/ROCm-DS/hipVS/blob/release/rocmds-25.10/cpp/include/cuvs/neighbors/cagra.hpp
  auto nnd_params =
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params(opt.k,
                                                                    dist_func);
  nnd_params.graph_degree              = opt.k;
  nnd_params.intermediate_graph_degree = opt.k;
  nnd_params.return_distances          = true;
  nnd_params.max_iterations            = opt.max_iterations;
  nnd_params.termination_threshold     = opt.delta;

  auto d_pstore_view = make_dev_matrix_view(points.data(), n_points, n_dims);

  saltatlas::rec_time().start("Build-knng");
  auto index = cuvs::neighbors::nn_descent::build(
      dev_res, nnd_params, make_const_matrix_view(d_pstore_view));
  saltatlas::rec_time().stop();
  SALTATLAS_HIP_CHECK(hipDeviceSynchronize());

  // show_index_score(index.distances()->data_handle(), n_points, opt.k,
  // false);

  print_time_table();
  saltatlas::rec_time().reset();

  if (!opt.output_path.empty()) {
    std::cout << "\nDump KNNG to " << opt.output_path << std::endl;
    const auto ids_view = saltatlas::solanet::apu_nn::matrix_view<id_type>(
        index.graph().data_handle(), n_points, opt.k);
    const auto dists_view = saltatlas::solanet::apu_nn::matrix_view<dist_type>(
        index.distances()->data_handle(), n_points, opt.k);
    dump_knng(ids_view, dists_view, opt.output_path, opt.dump_distance);
  }

  return 0;
}
