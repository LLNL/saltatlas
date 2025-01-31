// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of building k-NN index (KNN graph) in Metall
/// datastore. Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_advanced_index_build -p /path/to/points -f l2

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_advanced.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// Point Type
using point_type = saltatlas::pm_feature_vector<float>;

struct option_t {
  std::filesystem::path datastore_path;
  std::vector<id_t>     point_ids;
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.datastore_path.clear();
  help = false;

  int n;
  while ((n = ::getopt(argc, argv, "d:p:h")) != -1) {
    switch (n) {
      case 'd':
        opt.datastore_path = optarg;
        break;

      case 'p':  // comma separated list of point ids
      {
        std::string        point_ids_str = optarg;
        std::istringstream ss(point_ids_str);
        std::string        token;
        while (std::getline(ss, token, ',')) {
          opt.point_ids.push_back(std::stoi(token));
        }
        break;
      }

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  if (opt.datastore_path.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(cout_type &cout) {
  cout
      << "Usage: ./dnnd_advance_show_neighbors [options]\n"
         "Options:\n"
         "  -d <string>       The Metall datastore path\n"
         "  -p <string>       Comma separated list of point IDs (e.g., 0,2,5)\n"
         "  -h                Show this help message\n";
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  metall::logger::set_log_level(metall::logger::level_filter::verbose);

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
  if (!comm.rank0()) {
    // Only rank 0 request neighbors in this demo
    opt.point_ids.clear();
  }

  {
    saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::open_read_only,
                                                opt.datastore_path, comm);

    // Use the first index for demo
    const auto index_id = g.get_index_ids().front();
    comm.cout0() << "Index ID: " << index_id << std::endl;

    const auto ret =
        g.get_neighbors(index_id, opt.point_ids.begin(), opt.point_ids.end());
    for (const auto &item : ret) {
      const auto &pid       = item.first;
      const auto &neighbors = item.second;
      comm.cout0() << "Source point ID: " << pid << std::endl;
      for (const auto &n : neighbors) {
        comm.cout0() << "Neighbor ID: " << n.id << " Distance: " << n.distance
                     << ", " << std::endl;
      }
    }
    comm.cout0() << std::endl;

    // Demo get_neighbors_with_features
    const auto ret2 = g.get_neighbors_with_features(
        index_id, opt.point_ids.begin(), opt.point_ids.end());
    for (const auto &item : ret2) {
      const auto &pid       = item.first;
      const auto &neighbors = item.second.first;
      const auto &features  = item.second.second;
      comm.cout0() << "Source point ID: " << pid << std::endl;
      for (const auto &n : neighbors) {
        comm.cout0() << "Neighbor ID: " << n.id << " Distance: " << n.distance
                     << ", " << std::endl;
      }
      comm.cout0() << "Features: ";
      for (const auto &f : features) {
        comm.cout0() << saltatlas::to_string(f) << " ";
      }
    }

    comm.cf_barrier();
  }

  return 0;
}
