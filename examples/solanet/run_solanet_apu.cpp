// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SALTATLAS_SOLANET_APU
#define SALTATLAS_SOLANET_APU
#endif

#define SALTATLAS_SOLANET_APU_NN_SEARCH_MAX_BUF_SIZE 128
#define SALTATLAS_SOLANET_APU_NND_TEAM_SIZE 4
// #define SALTATLAS_SOLANET_APU_NN_QUERY_GRAPH_MAX_K 64

#include <stddef.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "solanet_common.hpp"

using namespace saltatlas;

struct options {
  std::string dataset_path;
  std::string dataset_format;
  size_t      k{0};
  size_t      nnd_k{0};
  double      rho{0.5};
  double      delta{0.0001};
  std::string dist_func{"l2"};
  size_t      max_degree{std::numeric_limits<size_t>::max()};
  int         refine_final_n_groups{2};
  int query_k{0};  // Number of neighbors to search for in the query phase. If
                   // 0, it will be set to k.
  int frontier_size{0};  // Search frontier size. If 0, it will be set to
                         // query_k * 2.
  bool        optimize{false};
  std::string knng_dump_dir;
  bool        dump_distance{false};
  bool        verbose{false};
  bool        development_verbose{false};

  template <typename out_stream_type>
  void show(out_stream_type& os) {
    os << "Options" << std::endl;
    os << "  Dataset: " << dataset_path << std::endl;
    os << "  Dataset format: " << dataset_format << std::endl;
    os << "  Distance function (l2 or cosine): " << dist_func << std::endl;
    os << "  k: " << k << std::endl;
    os << "  k for local kNNG construction: " << nnd_k << std::endl;
    os << "  #of groups after binary-tree-based refinement: "
       << refine_final_n_groups << std::endl;
    os << "  query_k: " << query_k << std::endl;
    os << "  frontier_size: " << frontier_size << std::endl;
    os << "  rho: " << rho << std::endl;
    os << "  delta: " << delta << std::endl;
    os << "  Optimize: " << optimize << std::endl;
    os << "  Max degree: " << max_degree << std::endl;
    os << "  Out dir: " << knng_dump_dir << std::endl;
    os << "  Dump distance: " << dump_distance << std::endl;
  }
};

bool parse_options(int argc, char* argv[], options& opt, bool& show_usage) {
  int p;
  while ((p = getopt(argc, argv, "i:p:f:k:a:n:F:r:d:G:DM:om:vVh")) != -1) {
    switch (p) {
      case 'i':
        opt.dataset_path = optarg;
        break;
      case 'p':
        opt.dataset_format = optarg;
        break;
      case 'f':
        opt.dist_func = optarg;
        break;
      case 'k':
        opt.k = std::stoi(optarg);
        break;
      case 'a':
        opt.nnd_k = std::stoi(optarg);
        break;
      case 'n':
        opt.query_k = std::stoi(optarg);
        break;
      case 'F':
        opt.frontier_size = std::stoi(optarg);
        break;
      case 'r':
        opt.rho = std::stod(optarg);
        break;
      case 'd':
        opt.delta = std::stod(optarg);
        break;
      case 'G':
        opt.knng_dump_dir = optarg;
        break;
      case 'D':
        opt.dump_distance = true;
        break;
      case 'M':
        opt.refine_final_n_groups = std::stoi(optarg);
        break;
      case 'o':
        opt.optimize = true;
        break;
      case 'm':
        opt.max_degree = std::stod(optarg);
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'V':
        opt.development_verbose = true;
        break;
      case 'h':
        show_usage = true;
        return true;
      default:
        return false;
    }
  }

  return true;
}

void show_usage(const char* prog_name) {
  std::cout << "Usage: " << prog_name
            << " -i <dataset_path> -p <dataset_format> -f <distance_function> "
               "-k <k> [options]"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -i <dataset_path>: Path to the dataset (required)"
            << std::endl;
  std::cout << "  -p <dataset_format>: Format of the dataset (required)"
            << std::endl;
  std::cout << "  -f <distance_function>: Distance function. l2 or ip (inner "
               "product), default: l2."
            << std::endl;
  std::cout << "  -k <k>: Number of neighbors final kNNG has (required)"
            << std::endl;
  std::cout << "  -a <nnd_k>: k value for initial local kNNG construction "
               "(default: 0, set to k)"
            << std::endl;
  std::cout << "  -n <query_k>: Number of neighbors to search for in the query "
               "phase. If 0, it will be set to k."
            << std::endl;
  std::cout << "  -F <frontier_size>: Search frontier size. If 0, it will be "
               "set to query_k * 2."
            << std::endl;
  std::cout
      << " -r <rho>: Sampling rate for candidate neighbors in local NN-Descent "
         "(default: 0.5)"
      << std::endl;
  std::cout << "  -d <delta>: Termination threshold for local NN-Descent "
               "(default: 0.0001)"
            << std::endl;
  std::cout
      << "  -m <max_degree>: Maximum degree after graph optimization. If not "
         "specified, it will be set to unlimited."
      << std::endl;
  std::cout
      << "  -M <refine_final_n_groups>: Number of groups to split the "
         "neighbors "
         "into for the final refinement step. If <= 1, no splitting is done."
      << std::endl;
  std::cout << "  -o: Optimize the KNNG by making it more symmetric and adding "
               "reciprocal edges (default: false)"
            << std::endl;
  std::cout << "  -G <knng_dump_dir>: If specified, dump the KNNG to the given "
               "directory. "
            << std::endl;
  std::cout
      << "  -D: Dump distances along with neighbor IDs when dumping KNNG. "
         "Only effective if -G is also specified."
      << std::endl;
  std::cout << "  -v: Verbose output (default: false)" << std::endl;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  signal(SIGSEGV, show_backtrace);
#endif

  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    mpi::communicator comm;
    options           opt;
    bool              show_usage = false;
    if (!parse_options(argc, argv, opt, show_usage)) {
      std::abort();
    }
    setup_spdlog(comm.cout0(), opt.verbose);
    spdlog::trace("SOLANET KNNG Builder");
    comm.barrier();

    if (provided < MPI_THREAD_FUNNELED) {
      comm.cerr0()
          << "The threading support level is lesser than that demanded."
          << std::endl;
      comm.abort();
    }
    comm.barrier();

    if (show_usage) {
      ::MPI_Finalize();
      std::_Exit(EXIT_SUCCESS);
    }
    comm.barrier();

    if (opt.verbose) {
      opt.show(comm.cout0());
      comm.show_mpi_info();
      show_omp_config(comm);
      show_common_config(comm);
    }
    comm.barrier();

    setup_apu(comm, opt.verbose);
    comm.barrier();

    solanet_engine::sparse_knng_type knng;
    {
      spdlog::trace("Build KNNG");
      solanet_engine::options solanet_opt{
          .k                     = opt.k,
          .nnd_k                 = opt.nnd_k,
          .delta                 = opt.delta,
          .rho                   = opt.rho,
          .dist_func             = opt.dist_func,
          .refine_final_n_groups = opt.refine_final_n_groups,
          .query_k               = opt.query_k,
          .frontier_size         = opt.frontier_size};
      spdlog::trace("Const SOLANET engine");
      solanet_engine solanet(solanet_opt, comm, opt.verbose);
      comm.barrier();

      std::vector<std::filesystem::path> paths{
          saltatlas::dndetail::find_file_paths(opt.dataset_path)};

      knng = solanet.build_index(paths, opt.dataset_format);
      comm.barrier();
    }

    if (opt.optimize) {
      comm.cout0() << "\n====================" << std::endl;
      comm.cout0() << "Optimize KNNG" << std::endl;
      comm.cout0() << "====================" << std::endl;
      comm.barrier();
      rec_time().start("Optimization");
      solanet::optimize_knng<id_type, dist_type>(
          knng, solanet::get_partitioner<id_type>(comm.size()), comm,
          opt.max_degree, opt.verbose);
      comm.barrier();
    }

#ifndef NDEBUG
    if (rec_time().num_running_timers() > 0) {
      comm.cerr0() << "Warning: some timers are still running." << std::endl;
    }
#endif

    if (opt.development_verbose) {
      comm.cout0() << "\nTime table (seconds):" << std::endl;
      comm.cout0() << "Name:\tMin,\tMax,\tMean,\tStd" << std::endl;
      const auto& time_table = rec_time().get_time_table();
      // For each entry, print min, mean, max, and standard deviation among
      // all MPI ranks.
      comm.cout0() << std::fixed << std::setprecision(2);
      for (const auto& entry : time_table) {
        std::vector<double> times(comm.size());
        comm.all_gather(entry.t, times.data());
        const auto [min, mean, max, std] = saltatlas::detail::get_stats(times);
        for (std::size_t i = 0; i < entry.depth; ++i) {
          comm.cout0() << "  ";
        }
        comm.cout0() << entry.name << ":\t" << min << ",\t" << mean << ",\t"
                     << max << ",\t" << std << std::endl;
      }
      comm.barrier();
    }

    if (!opt.knng_dump_dir.empty()) {
      comm.cout0() << "\n====================" << std::endl;
      spdlog::trace("Dump KNNG");
      comm.cout0() << "====================" << std::endl;
      std::error_code ec;
      std::filesystem::create_directories(opt.knng_dump_dir, ec);
      comm.barrier();
      solanet::dump_knng(opt.knng_dump_dir, knng, comm, opt.dump_distance);
      comm.barrier();
      spdlog::trace("Dumped to {}", opt.knng_dump_dir);
    }
    comm.barrier();

    comm.cout0() << "\n====================" << std::endl;
    comm.cout0() << "Finished SOLANET" << std::endl;
    comm.cout0() << "====================" << std::endl;
    comm.barrier();
  }
  ::MPI_Finalize();
  std::_Exit(EXIT_SUCCESS);
}
