// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

/// \brief A simple example of ANN search using DNND's advanced API with string
/// point IDs.

#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
  std::filesystem::path datastore_path;
  std::filesystem::path out_datastore_path;
};

bool parse_options(int argc, char **argv, option_t &opt, bool &help,
                   std::string &error_message) {
  opt.datastore_path.clear();
  opt.out_datastore_path.clear();
  help = false;
  error_message.clear();

  opterr = 0;
  int n;
  while ((n = ::getopt(argc, argv, "d:o:h")) != -1) {
    switch (n) {
      case 'd':
        opt.datastore_path = optarg;
        break;

      case 'o':
        opt.out_datastore_path = optarg;
        break;

      case 'h':
        help = true;
        return true;

      case ':':
        error_message = std::string("Missing argument for option -") +
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
  if (opt.out_datastore_path.empty()) {
    missing_required_options.push_back("-o <out_datastore_path>");
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
          "  -o <out_datastore_path> Path to the main Metall datastore\n"
          "\nOptional options:\n"
          "<prefix>-<rank>\n"
          "  -h                       Show this help message\n";
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  option_t    opt;
  bool        help{false};
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

  {
    dnnd_type dnnd(saltatlas::open_read_only, opt.datastore_path, comm);
    dnnd.aggregate_datastore(opt.out_datastore_path);
  }

  return 0;
}
