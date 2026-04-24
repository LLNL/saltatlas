// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of ANN search using DNND's advanced API with string
/// point IDs.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <ygm/comm.hpp>

#include <saltatlas/dnnd/dnnd_adv.hpp>

using id_type    = saltatlas::pm_str_id_type;
using point_type = saltatlas::pm_feature_vector<float>;
using dist_type  = float;
using dnnd_type =
    saltatlas::dnnd_adv<id_type, point_type, dist_type, saltatlas::str_hash<>>;

struct option_t {
  std::filesystem::path datastore_path;
  std::string           distance_name;
  std::filesystem::path query_file_path;
  std::filesystem::path out_file_path;
  int                   query_n{0};  // #of neighbor points to search
};

template <typename neighbor_store_type>
bool dump_local_query_results(const neighbor_store_type   &results,
                              const std::filesystem::path &out_file_path,
                              ygm::comm                   &comm);

bool parse_options(int argc, char **argv, option_t &opt, bool &help,
                   std::string &error_message) {
  opt.datastore_path.clear();
  opt.distance_name.clear();
  opt.query_file_path.clear();
  opt.out_file_path.clear();
  opt.query_n = 0;
  help        = false;
  error_message.clear();

  opterr = 0;
  int n;
  while ((n = ::getopt(argc, argv, ":d:f:q:n:o:h")) != -1) {
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
        try {
          opt.query_n = std::stoi(optarg);
        } catch (...) {
          error_message = "Option -n must be a positive integer.";
          return false;
        }
        if (opt.query_n <= 0) {
          error_message = "Option -n must be greater than 0.";
          return false;
        }
        break;

      case 'o':
        opt.out_file_path = optarg;
        break;

      case 'h':
        help = true;
        return true;

      case ':':
        error_message =
            std::string("Missing argument for option -") +
            static_cast<char>(optopt) + ".";
        return false;

      case '?':
        if (optopt == 0) {
          error_message = "Unknown option.";
        } else {
          error_message =
              std::string("Unknown option -") + static_cast<char>(optopt) + ".";
        }
        return false;

      default:
        return false;
    }
  }

  std::vector<std::string> missing_required_options;
  if (opt.datastore_path.empty()) {
    missing_required_options.push_back("-d <datastore_path>");
  }
  if (opt.distance_name.empty()) {
    missing_required_options.push_back("-f <distance_function>");
  }
  if (opt.query_file_path.empty()) {
    missing_required_options.push_back("-q <query_file_path>");
  }
  if (opt.query_n <= 0) {
    missing_required_options.push_back("-n <num_neighbors>");
  }

  if (!missing_required_options.empty()) {
    std::ostringstream oss;
    oss << "Missing required option(s): ";
    for (std::size_t i = 0; i < missing_required_options.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << missing_required_options[i];
    }
    error_message = oss.str();
    return false;
  }

  return true;
}

template <typename cout_type>
void show_help(const std::string &exe_name, cout_type &cout) {
  cout << "Usage: " << exe_name
       << " -d <datastore_path> -f <distance_function> "
          "-q <query_file_path> -n <num_neighbors> "
          "[-o <output_path_prefix>] [-h]\n"
          "\nRequired options:\n"
          "  -d <datastore_path>      Path to the Metall datastore\n"
          "  -f <distance_function>   Distance function name\n"
          "  -q <query_file_path>     Path to query file\n"
          "  -n <num_neighbors>       Number of neighbors to retrieve (> 0)\n"
          "\nOptional options:\n"
          "  -o <output_path_prefix>  Dump local query results to "
          "<prefix>-<rank>\n"
          "  -h                       Show this help message\n";
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  option_t opt;
  bool     help{false};
  std::string option_error;
  if (!parse_options(argc, argv, opt, help, option_error)) {
    if (!option_error.empty()) {
      comm.cerr0() << "Invalid options: " << option_error << std::endl;
    } else {
      comm.cerr0() << "Invalid options." << std::endl;
    }
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
      saltatlas::distance::distance_function<point_type, dist_type>(
          opt.distance_name);
  {
    dnnd_type dnnd(saltatlas::open_read_only, opt.datastore_path, comm);

    // Run queries on the first index.
    const auto index_id = dnnd.get_index_ids().front();

    // Run queries
    comm.cout0() << "Run queries on index " << index_id << std::endl;
    const auto ret = dnnd.query(index_id, distance_func, queries.begin(),
                                queries.end(), opt.query_n);

    const bool dump_ok = dump_local_query_results(ret, opt.out_file_path, comm);
    if (!dump_ok && comm.rank0()) {
      comm.cerr0() << "Failed to dump query results on one or more ranks."
                   << std::endl;
    }
    if (!opt.out_file_path.empty() && comm.rank0()) {
      comm.cout0() << "Dumped local query results to "
                   << opt.out_file_path.string() << "-<rank>" << std::endl;
    }

    comm.cf_barrier();
  }

  return 0;
}

template <typename neighbor_store_type>
bool dump_local_query_results(const neighbor_store_type   &results,
                              const std::filesystem::path &out_file_path,
                              ygm::comm                   &comm) {
  if (out_file_path.empty()) {
    return true;
  }

  const auto parent_path = out_file_path.parent_path();
  if (!parent_path.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent_path, ec);
    if (ec) {
      std::cerr << "Rank " << comm.rank() << ": Failed to create directory "
                << parent_path << " (" << ec.message() << ")" << std::endl;
      return false;
    }
  }

  const auto rank_out_file =
      out_file_path.string() + "-" + std::to_string(comm.rank());
  std::ofstream ofs(rank_out_file);
  if (!ofs.is_open()) {
    std::cerr << "Rank " << comm.rank() << ": Failed to open output file "
              << rank_out_file << std::endl;
    return false;
  }

  // Keep the same layout as utility::dump_neighbors:
  // first block = neighbor IDs, second block = neighbor distances.
  for (const auto &neighbors : results) {
    for (std::size_t k = 0; k < neighbors.size(); ++k) {
      if (k > 0) ofs << "\t";
      ofs << neighbors[k].id;
    }
    ofs << "\n";
    for (std::size_t k = 0; k < neighbors.size(); ++k) {
      if (k > 0) ofs << "\t";
      ofs << neighbors[k].distance;
    }
    ofs << "\n";
  }

  return true;
}
