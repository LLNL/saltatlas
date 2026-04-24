// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of building k-NN index (KNN graph) in Metall
/// datastore with string point IDs.

#include <mpi.h>

#include <unistd.h>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

using id_type    = saltatlas::pm_str_id_type;
using point_type = saltatlas::pm_feature_vector<float>;
using dist_type  = float;
using dnnd_type =
    saltatlas::dnnd_adv<id_type, point_type, dist_type, saltatlas::str_hash<>>;

struct option_t {
  int                                index_k{4};
  std::string                        distance_name{"l2"};
  std::vector<std::filesystem::path> point_file_names{};
  std::string                        point_file_format;
  std::filesystem::path              datastore_path{"/tmp/dnnd-index"};

  // Optional arguments
  bool                  dump_index{false};
  std::filesystem::path index_dump_path{"/tmp/dnnd-index-dump"};
};

bool parse_positive_int(const char *arg, int &value) {
  if (arg == nullptr) {
    return false;
  }

  int        parsed = 0;
  const auto begin  = arg;
  const auto end    = arg + std::strlen(arg);
  const auto result = std::from_chars(begin, end, parsed);

  if (result.ec != std::errc{} || result.ptr != end || parsed <= 0) {
    return false;
  }

  value = parsed;
  return true;
}

bool parse_options(int argc, char **argv, option_t &opt, bool &help,
                   std::string &error_message) {
  opt.index_k = 0;
  opt.distance_name.clear();
  opt.point_file_names.clear();
  opt.point_file_format.clear();
  opt.datastore_path.clear();
  help          = false;
  error_message = "";

  bool has_k = false;
  bool has_f = false;
  bool has_p = false;
  bool has_d = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:f:p:d:Do:h")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::stoi(optarg);
        has_k       = true;
        break;

      case 'f':
        opt.distance_name = optarg;
        has_f             = true;
        break;

      case 'p':
        opt.point_file_format = optarg;
        has_p                 = true;
        break;

      case 'd':
        opt.datastore_path = optarg;
        has_d              = true;
        break;

      case 'D':
        opt.dump_index = true;
        break;

      case 'o':
        opt.index_dump_path = optarg;
        break;

      case 'h':
        help = true;
        return true;

      default:
        if (optopt != 0) {
          std::stringstream ss;
          ss << "Unknown option '-" << static_cast<char>(optopt) << "'";
          error_message = ss.str();
        } else {
          error_message = "Invalid command line arguments";
        }
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  std::vector<std::string> missing;
  if (!has_k || opt.index_k <= 0) {
    missing.emplace_back("-k <int>");
  }
  if (!has_f || opt.distance_name.empty()) {
    missing.emplace_back("-f <string>");
  }
  if (!has_p || opt.point_file_format.empty()) {
    missing.emplace_back("-p <string>");
  }
  if (!has_d || opt.datastore_path.empty()) {
    missing.emplace_back("-d <string>");
  }
  if (opt.point_file_names.empty()) {
    missing.emplace_back("point_files...");
  }

  if (!missing.empty()) {
    std::stringstream ss;
    ss << "Missing required argument(s): ";
    for (std::size_t i = 0; i < missing.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << missing[i];
    }
    error_message = ss.str();
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
          "  -D                Dump index to text file\n"
          "  -o <string>       Index dump file path\n"
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

    if (opt.dump_index) {
      comm.cout0() << "Dumping index to file: " << opt.index_dump_path
                   << std::endl;
      dnnd.dump_index(index_id, opt.index_dump_path);
    }
  } catch (const std::exception &e) {
    comm.cerr0() << "Index build failed: " << e.what() << std::endl;
    MPI_Abort(comm.get_mpi_comm(), EXIT_FAILURE);
    std::abort();
  }

  return 0;
}
