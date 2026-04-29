// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#define SALTATLAS_SOLANET_APU_NND_TEAM_SIZE 8

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include <saltatlas/dnnd/distance.hpp>
#include <saltatlas/dnnd/utility.hpp>
#include <saltatlas/shm_knng_query/data_reader.hpp>
#include <saltatlas/solanet/detail/apu_nn/matrix.hpp>
#include <saltatlas/solanet/detail/apu_nn/memory.hpp>
#include <saltatlas/solanet/detail/apu_nn/nndescent.hpp>
#include <saltatlas/solanet/detail/apu_nn/search_knng_builder.hpp>
#include <saltatlas/solanet/detail/nn_index_view.hpp>
#include <saltatlas/solanet/singleton_time_recorder.hpp>

#include "l2_normalize_points.hpp"
#include "solanet_shm_common.hpp"

struct options {
  std::filesystem::path point_files_path;
  std::string           point_file_format;
  std::string           distance_function;
  size_t                k{0};
  double                rho{0.5};
  double                delta{0.0001};
  int                   max_iterations{100};
  bool                  optimize{false};
  std::string           output_path{};
  bool                  dump_distance{false};

  /// Show the option values
  void show() const {
    std::cout << "Option:" << std::endl;
    std::cout << "point_files_path: " << point_files_path << std::endl;
    std::cout << "point_file_format: " << point_file_format << std::endl;
    std::cout << "distance_function: " << distance_function << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "rho: " << rho << std::endl;
    std::cout << "delta: " << delta << std::endl;
    std::cout << "max_iterations: " << max_iterations << std::endl;
    std::cout << "optimize: " << optimize << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "dump_distance: " << dump_distance << std::endl;
  }
};

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:d:r:m:oG:Dh")) != -1) {
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
        opt.k = std::stoi(optarg);
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
      case 'o':
        opt.optimize = true;
        break;
      case 'G':
        opt.output_path = optarg;
        break;
      case 'D':
        opt.dump_distance = true;
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
               "product), default: l2."
            << std::endl;
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
  std::cout << "  -o: Optimize the KNNG by making it more symmetric and adding "
               "reciprocal edges (default: false)"
            << std::endl;
  std::cout
      << "  -G <output_path>: If specified, dump the KNNG to the given path. "
         "Distance will be dumped if -D is also specified."
      << std::endl;
  std::cout
      << "  -D: Dump distances along with neighbor IDs when dumping KNNG. "
         "Only effective if -G is also specified."
      << std::endl;
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
  std::printf("Using device %d: %s (gfx: %s)\n", dev, prop.name,
              prop.gcnArchName);

  std::cout << "\nLoad point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_path);
  auto points = saltatlas::load_points<
      id_type, id_type, fe_type, e2i_id_map_type,
      saltatlas::solanet::apu_nn::hip_allocator<fe_type>>(
      point_file_paths, opt.point_file_format);
  std::cout << "Number of points: " << points.num_points() << std::endl;
  std::cout << "Number of dimensions: " << points.num_dimensions() << std::endl;

  if (opt.distance_function == "ip") {
    std::cout << "!!! L2 normalizing points for inner product search... !!"
              << std::endl;
    l2_normalize_points(points.data(), points.num_points(),
                        points.num_dimensions());
  }

  {
    std::cout << "\nBuild KNNG" << std::endl;
    saltatlas::solanet::apu_nn::matrix_view<fe_type> pstore_view(
        points.data(), points.num_points(), points.num_dimensions());
    saltatlas::rec_time().start("Build-knng");
    auto [knn_ids, knn_dists] =
        saltatlas::solanet::apu_nn::build_index<id_type, fe_type, dist_type>(
            pstore_view, opt.distance_function, opt.k, opt.rho, opt.delta,
            std::random_device{}(), opt.max_iterations);
    saltatlas::rec_time().stop();  // Build-knng

    if (opt.optimize) {
      std::cout << "\nOptimize KNNG" << std::endl;
      saltatlas::rec_time().start("Optimization");
      saltatlas::solanet::apu_nn::make_optimized_query_graph<id_type,
                                                             dist_type>(
          knn_ids.get_view(), knn_dists.get_view(), knn_ids.get_view());
      saltatlas::rec_time().stop();  // Optimization
    }
    print_time_table();
    saltatlas::rec_time().reset();

    if (!opt.output_path.empty()) {
      std::cout << "\nDump KNNG to " << opt.output_path << std::endl;
      if (opt.optimize) {
        opt.dump_distance = false;
        std::cout << "Note: Dumping distance is not supported for optimized "
                     "KNNG. Distance will not be dumped."
                  << std::endl;
      }
      dump_knng(knn_ids.get_view(), knn_dists.get_view(), opt.output_path,
                opt.dump_distance);
    }
  }

  return 0;
}
