// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of building k-NN index (KNN graph) in Metall
/// datastore. Usage:
///     cd build
///     mpirun -n 2 ./example/dnnd_adv_index_build -p /path/to/points -f l2

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

// Point ID type
using id_t   = uint32_t;
using dist_t = double;

// Point Type
using point_type = saltatlas::pm_feature_vector<float>;

struct option_t {
  std::filesystem::path datastore_path;
  std::string           distance_name;
  std::filesystem::path query_file_path;
  int                   query_n;  // #of neighbor points to search
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.datastore_path.clear();
  opt.distance_name.clear();
  opt.query_file_path.clear();
  opt.query_n = 0;
  help        = false;

  int n;
  while ((n = ::getopt(argc, argv, "d:f:q:n:h")) != -1) {
    switch (n) {
      case 'd':
        opt.datastore_path = optarg;
        break;

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'q':
        opt.query_file_path = optarg;
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
      opt.query_file_path.empty() || opt.query_n <= 0) {
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(const std::string &exe_name, cout_type &cout) {
  cout << "Usage: " << exe_name
       << " [options]\n"
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
    show_help(argv[0], comm.cerr0());
    return EXIT_FAILURE;
  }
  if (help) {
    show_help(argv[0], comm.cout0());
    return 0;
  }

  std::vector<point_type> queries;
  saltatlas::read_query(opt.query_file_path, queries, comm);

  const auto distance_func =
      saltatlas::distance::distance_function<point_type, dist_t>(
          opt.distance_name);
  {
    saltatlas::dnnd_adv<id_t, point_type, dist_t> g(saltatlas::open_read_only,
                                                    opt.datastore_path, comm);
    const auto                                index_ids = g.get_index_ids();

    // Run queries
    {
      for (const auto index_id : index_ids) {
        comm.cout0() << "Run queries on index " << index_id << std::endl;
        const auto ret = g.query(index_id, distance_func, queries.begin(),
                                 queries.end(), opt.query_n);
        for (std::size_t i = 0; i < ret.size(); ++i) {
          comm.cout0() << "Query " << i << ":\n";
          for (const auto &neighbor : ret[i]) {
            comm.cout0() << "  " << neighbor << std::endl;
          }
        }
      }
    }
    comm.cf_barrier();

    // Run queries and receive neighbor features
    {
      for (const auto index_id : index_ids) {
        comm.cout0() << "\nRun queries on index " << index_id << std::endl;
        const auto ret =
            g.query_with_features(index_id, distance_func, queries.begin(),
                                  queries.end(), opt.query_n);
        const auto &neighbors = ret.first;
        const auto &features  = ret.second;
        for (std::size_t qi = 0; qi < queries.size(); ++qi) {
          comm.cout0() << "Query " << qi << ":\n";
          for (int ni = 0; ni < neighbors[qi].size(); ++ni) {
            comm.cout0() << neighbors[qi][ni] << ", feature = "
                         << saltatlas::to_string(features[qi][ni]) << std::endl;
          }
          comm.cout0() << std::endl;
        }
      }
    }
  }

  return 0;
}
