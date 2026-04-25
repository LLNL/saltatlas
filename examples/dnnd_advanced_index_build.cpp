// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of building k-NN index (KNN graph) in Metall
/// datastore with string point IDs.

#include <mpi.h>

#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

#if defined(SALTATLAS_USE_STRING_ID)
using id_type = saltatlas::pm_str_id_type;
#else
using id_type = uint32_t;
#endif
using point_type = saltatlas::pm_feature_vector<float>;
using dist_type  = float;
using dnnd_type =
    saltatlas::dnnd_adv<id_type, point_type, dist_type, saltatlas::str_hash<>>;

struct option_t {
  int                                index_k{0};
  std::string                        distance_name{};
  std::vector<std::filesystem::path> point_file_names{};
  std::string                        point_file_format{};
  std::filesystem::path              datastore_path{};

  // Optional arguments
  std::filesystem::path index_dump_path;
  std::filesystem::path external_id_map_dump_path;
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help,
                   std::string &error_message) {
  opt  = option_t{};
  help = false;
  error_message.clear();
  ::optind = 1;
  ::opterr = 0;

  int n;
  while ((n = ::getopt(argc, argv, "k:f:p:d:G:M:h")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::atoi(optarg);
        break;

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'd':
        opt.datastore_path = optarg;
        break;

      case 'G':
        opt.index_dump_path = optarg;
        break;

      case 'M':
        opt.external_id_map_dump_path = optarg;
        break;

      case 'h':
        help = true;
        return true;

      default:
        error_message = "Invalid command line arguments";
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.index_k <= 0 || opt.distance_name.empty() ||
      opt.point_file_format.empty() || opt.datastore_path.empty() ||
      opt.point_file_names.empty()) {
    error_message = "Missing required command line arguments";
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(const std::string &exe_name, cout_type &cout) {
  cout << "Usage:\n"
       << "  " << exe_name
       << " -k <int> -f <string> -p <string> -d <string> point_files...\n"
          "\n"
          "Required:\n"
          "  -k <int>          Number of neighbors to build per point\n"
          "  -f <string>       Distance function name (e.g., l2)\n"
          "  -p <string>       Input point file format (e.g., wsv-id)\n"
          "  -d <string>       Metall datastore path to create\n"
          "  point_files...    One or more input point files matching -p "
          "format\n"
          "\n"
          "Optional:\n"
          "  -G <string>       Index dump file path\n"
          "  -M <string>       External-ID to internal-ID map file path\n"
          "  -h                Show this help message\n"
          "\n"
          "Example:\n"
          "  mpirun -n 2 "
       << exe_name
       << " -k 32 -f l2 -p wsv-id -d /tmp/dnnd-index point_0.wsv point_1.wsv\n";
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  option_t    opt;
  bool        help{false};
  std::string parse_error;
  if (!parse_options(argc, argv, opt, help, parse_error)) {
    if (!parse_error.empty()) {
      comm.cerr0() << parse_error << std::endl;
    } else {
      comm.cerr0() << "Invalid command line arguments" << std::endl;
    }
    show_help(argv[0], comm.cerr0());
    return EXIT_FAILURE;
  }
  if (help) {
    show_help(argv[0], comm.cout0());
    return 0;
  }

  if (comm.rank0()) {
    std::error_code ec;
    std::filesystem::remove_all(opt.datastore_path, ec);
    if (ec) {
      comm.cerr0() << "Failed to clean datastore path '" << opt.datastore_path
                   << "': " << ec.message() << std::endl;
      return EXIT_FAILURE;
    }
  }
  comm.barrier();

  try {
    dnnd_type dnnd(saltatlas::create_only, opt.datastore_path, comm);

    dnnd.load_points(opt.point_file_names.begin(), opt.point_file_names.end(),
                     opt.point_file_format);

    const auto distance_func =
        saltatlas::distance::distance_function<point_type, dist_type>(
            opt.distance_name);

    comm.cout0() << "Building index" << std::endl;
    const auto index_id = dnnd.build(distance_func, opt.index_k);

    comm.cout0() << "Optimizing index" << std::endl;
    dnnd.optimize(index_id, distance_func);

    if (!opt.index_dump_path.empty()) {
      comm.cout0() << "Dumping index to file: " << opt.index_dump_path
                   << std::endl;
      const bool dump_distance = false;
      dnnd.dump_index(index_id, opt.index_dump_path, dump_distance);
    }

    if (!opt.external_id_map_dump_path.empty()) {
      comm.cout0() << "Dumping external-ID-to-internal-ID map to file: "
                   << opt.external_id_map_dump_path << std::endl;
      if (!dnnd.dump_external_id_map(opt.external_id_map_dump_path)) {
        return EXIT_FAILURE;
      }
    }
  } catch (const std::exception &e) {
    comm.cerr0() << "Index build failed: " << e.what() << std::endl;
    MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    std::abort();
  }

  return 0;
}
