// Copyright 2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <ygm/comm.hpp>

#include <saltatlas/common/data_reader.hpp>
#include <saltatlas/common/point_store.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>
#include <saltatlas/dnnd/utility.hpp>

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using point_type = saltatlas::feature_vector<fe_type>;

struct option {
  std::filesystem::path points_dir;                   // i
  std::string           point_file_format{"wsv-id"};  // p
  std::string           output_dir;                   // o
  int                   c       = 0;                  // c
  size_t                id_base = 0;                  // B
};

// use getopt to parse command line options
bool parse_options(int argc, char** argv, option& opt) {
  int opt_char;
  while ((opt_char = ::getopt(argc, argv, "i:p:o:c:B:")) != -1) {
    switch (opt_char) {
      case 'i':
        opt.points_dir = std::filesystem::path(optarg);
        break;
      case 'p':
        opt.point_file_format = optarg;
        break;
      case 'c':
        opt.c = std::stoi(optarg);
        break;
      case 'o':
        opt.output_dir = optarg;
        break;
      case 'B':
        opt.id_base = std::stoul(optarg);
        break;
      default:
        return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  ygm::comm comm(&argc, &argv);

  option opt;
  parse_options(argc, argv, opt);
  if (opt.points_dir.empty()) {
    comm.cerr0() << "Points directory is not specified." << std::endl;
    return EXIT_FAILURE;
  }
  if (opt.point_file_format.empty()) {
    comm.cerr0() << "Point file format is not specified." << std::endl;
    return EXIT_FAILURE;
  }
  if (opt.c < 0) {
    comm.cerr0() << "Column index must be non-negative." << std::endl;
    return EXIT_FAILURE;
  }
  if (opt.output_dir.empty()) {
    comm.cerr0() << "Output directory is not specified." << std::endl;
    return EXIT_FAILURE;
  }

  if (opt.point_file_format != "wsv-id") {
    comm.cerr0() << "Unsupported point file format: " << opt.point_file_format
                 << std::endl;
    return EXIT_FAILURE;
  }

  saltatlas::point_store<id_type, point_type> point_store;
  comm.cout0() << "\n<<Read Points>>" << std::endl;
  {
    const auto paths = saltatlas::utility::find_file_paths(opt.points_dir);
    saltatlas::read_points<id_type, point_type>(
        paths, opt.point_file_format, true,
        [&](const id_type& id) { return id % comm.size(); }, point_store, comm);
  }
  comm.cout0() << "Total points read: " << ygm::sum(point_store.size(), comm)
               << std::endl;

  // Find min and max value at column 'opt.c' for normalization.
  fe_type col_min = std::numeric_limits<fe_type>::max();
  fe_type col_max = std::numeric_limits<fe_type>::min();
  for (const auto& [id, point] : point_store) {
    col_min = std::min(col_min, point.at(opt.c));
    col_max = std::max(col_max, point.at(opt.c));
  }
  comm.cf_barrier();
  col_min = ygm::min(col_min, comm);
  col_max = ygm::max(col_max, comm);
  comm.cout0() << "Min value at column " << opt.c << ": " << col_min
               << std::endl;
  comm.cout0() << "Max value at column " << opt.c << ": " << col_max
               << std::endl;

  // Make output directory if it does not exist
  comm.cout0() << "Output directory: " << opt.output_dir << std::endl;
  if (comm.rank0()) {
    std::filesystem::create_directories(opt.output_dir);
  }
  comm.cf_barrier();

  if (!point_store.empty()) {
    std::string output_file =
        opt.output_dir + "/points-" + std::to_string(comm.rank()) + ".txt";
    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
      std::cerr << "Failed to open output file: " << output_file << std::endl;
      return EXIT_FAILURE;
    }
    for (const auto& [id, point] : point_store) {
      if (opt.point_file_format.find("id") != std::string::npos) {
        ofs << (size_t(id) + opt.id_base) << " ";
      }
      for (int i = 0; i < point.size(); ++i) {
        fe_type value = point.at(i);
        if (i == opt.c) {
          value = (value - col_min) + col_max + 0.1 * (col_max - col_min);
        }
        ofs << value << " ";
      }
      ofs << "\n";
    }
    ofs.close();
    if (!ofs) {
      std::cerr << "Failed to write to output file: " << output_file
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  comm.cf_barrier();
  comm.cout0() << "Finished writing output." << std::endl;

  return EXIT_SUCCESS;
}
