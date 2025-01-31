// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of building k-NN index (KNN graph) in Metall
/// datastore. Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_advanced_index_build -p /path/to/points -f l2

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_advanced.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// Point Type
using point_type = saltatlas::pm_feature_vector<float>;

struct option_t {
  int                                index_k;
  std::string                        distance_name;
  std::vector<std::filesystem::path> point_file_names;
  std::string                        point_file_format;
  std::filesystem::path              datastore_path;
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.index_k = 0;
  opt.distance_name.clear();
  opt.point_file_names.clear();
  opt.point_file_format.clear();
  opt.datastore_path.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:f:p:d:h")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::stoi(optarg);
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

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.index_k <= 0) {
    return false;
  }

  if (opt.distance_name.empty() || opt.point_file_format.empty() ||
      opt.point_file_names.empty() || opt.datastore_path.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(cout_type &cout) {
  cout << "Usage: ./dnnd_advanced_index_build [options] point_files...\n"
          "Options:\n"
          "  -k <int>          The number of neighbors to build the index\n"
          "  -f <string>       The distance function name\n"
          "  -p <string>       The point file format\n"
          "  -d <string>       The Metall datastore path\n"
          "  -h                Show this help message\n";
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  option_t opt;
  bool     help{false};
  if (!parse_options(argc, argv, opt, help)) {
    comm.cerr0() << "Invalid option" << std::endl;
    show_help(comm.cerr0());
    return EXIT_FAILURE;
  }
  if (help) {
    show_help(comm.cout0());
    return 0;
  }

  std::error_code ec;
  std::filesystem::remove_all(opt.datastore_path, ec);
  comm.cf_barrier();
  {
    saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::create_only,
                                                opt.datastore_path, comm);

    g.load_points(opt.point_file_names.begin(), opt.point_file_names.end(),
                  opt.point_file_format);

    const auto distance_func =
        saltatlas::distance::distance_function<point_type, dist_t>(
            opt.distance_name);

    comm.cout0() << "Building index" << std::endl;
    const auto index_id = g.build(distance_func, opt.index_k);

    comm.cout0() << "Optimizing index" << std::endl;
    g.optimize(index_id, distance_func);
  }

  return 0;
}
