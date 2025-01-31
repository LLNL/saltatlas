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
  std::string           distance_name;
  int                   query_n;  // #of neighbor points to search
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.datastore_path.clear();
  opt.point_ids.clear();
  opt.distance_name.clear();
  opt.query_n = 0;
  help        = false;

  int n;
  while ((n = ::getopt(argc, argv, "d:p:f:n:h")) != -1) {
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

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'n':
        opt.query_n = std::stoi(optarg);
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  if (opt.datastore_path.empty() || opt.distance_name.empty() ||
      opt.point_ids.empty()) {
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(cout_type &cout) {
  cout << "Usage: ./dnnd_advance_show_neighbors [options]\n"
          "Options:\n"
          "  -d <string>       The Metall datastore path\n"
          "  -f <string>       The distance function\n"
          "  -p <string>       Comma separated list of query source point IDs "
          "(e.g., 0,2,5)\n"
          "  -n <int>          The number of neighbor points to search\n"
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

  {
    saltatlas::dnnd<id_t, point_type, dist_t> g(saltatlas::open_read_only,
                                                opt.datastore_path, comm);
    const auto                                distance_func =
        saltatlas::distance::distance_function<point_type, dist_t>(
            opt.distance_name);
    const auto index_ids = g.get_index_ids();
    for (const auto index_id : index_ids) {
      comm.cout0() << "Run queries on index: " << index_id << std::endl;
      const auto ret = g.query(index_id, distance_func, opt.point_ids.begin(),
                               opt.point_ids.end(), opt.query_n);
      for (int qid = 0; qid < ret.size(); ++qid) {
        const auto &pid       = opt.point_ids[qid];
        const auto &neighbors = ret[qid];
        comm.cout0() << "Query source point ID: " << pid << std::endl;
        for (const auto &n : neighbors) {
          comm.cout0() << "Neighbor ID: " << n.id << " Distance: " << n.distance
                       << ", " << std::endl;
        }
      }
      comm.cout0() << "\n" << std::endl;
    }
    comm.cf_barrier();
  }

  return 0;
}
