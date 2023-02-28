// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

// Usage example:
// Two input files. Split each file into 10 files equally.
// OMP_NUM_THREADS=8 ./convert_wsv2wsv_id -o /path/to/wsv-id/prefix -n 10 \
// /path/to/wsv/file-0 /path/to/wsv/file-1

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <metall/detail/utilities.hpp>
#include <metall/utility/open_mp.hpp>

inline bool parse_options(int argc, char** argv,
                          std::vector<std::string>& inputs,
                          std::string&              out_file_prefix,
                          std::size_t&              num_splits) {
  inputs.clear();
  out_file_prefix.clear();
  num_splits = 1;

  int n;
  while ((n = ::getopt(argc, argv, "o:n:")) != -1) {
    switch (n) {
      case 'o':
        out_file_prefix = optarg;
        break;

      case 'n':
        num_splits = std::stoul(optarg);
        break;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    inputs.emplace_back(argv[index]);
  }

  return true;
}

int main(int argc, char** argv) {
  std::vector<std::string> inputs;  // input wsv files (must be sorted).
  std::string              out_file_prefix;  // output file prefix.
  std::size_t              num_splits;       // #of splits to generate per input file.
  parse_options(argc, argv, inputs, out_file_prefix, num_splits);

  std::vector<std::size_t> num_points(inputs.size(), 0);
  OMP_DIRECTIVE(parallel for)
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::ifstream ifs(inputs[i]);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open: " << inputs[i] << std::endl;
      std::abort();
    }

    for (std::string buf; std::getline(ifs, buf);) {
      ++num_points[i];
    }
  }
  const auto num_total_points =
      std::accumulate(num_points.begin(), num_points.end(), (std::size_t)0);
  std::cout << "#of total points\t" << num_total_points << std::endl;

  // Calculate ID offset of each input file.
  std::vector<std::size_t> offsets(inputs.size() + 1, 0);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    for (std::size_t j = i + 1; j < offsets.size(); ++j) {
      offsets[j] += num_points[i];
    }
  }
  if (offsets.back() != num_total_points) {
    std::cerr << "Abort at " << __LINE__ << std::endl;
    std::abort();
  }

  // Generate output file, adding ID in the first column and splitting into
  // smaller files.
  OMP_DIRECTIVE(parallel for)
  for (std::size_t if_no = 0; if_no < inputs.size(); ++if_no) {
    std::ifstream ifs(inputs[if_no]);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open: " << inputs[if_no] << std::endl;
      std::abort();
    }

    std::size_t id = offsets[if_no];
    for (std::size_t chunk_no = 0; chunk_no < num_splits; ++chunk_no) {
      const auto new_file_no = num_splits * if_no + chunk_no;
      const auto new_file_name =
          out_file_prefix + "-" + std::to_string(new_file_no) + ".txt";
      std::ofstream ofs(new_file_name);
      if (!ofs.is_open()) {
        std::cerr << "Failed to create " << new_file_name << std::endl;
        std::abort();
      }

      const auto range = metall::mtlldetail::partial_range(
          num_points[if_no], chunk_no, num_splits);
      for (std::size_t i = 0; i < range.second - range.first; ++i) {
        std::string buf;
        if (!std::getline(ifs, buf)) {
          std::cerr << "Failed to read from " << inputs[if_no] << std::endl;
          std::abort();
        }
        ofs << id << " " << buf << "\n";
        ++id;
      }

      ofs.close();
      if (!ofs) {
        std::cerr << "Failed to write to " << new_file_name << std::endl;
        std::abort();
      }
    }
    if (id != offsets[if_no + 1]) {
      std::cerr << "Abort at " << __LINE__ << std::endl;
      std::abort();
    }
  }

  std::cout << "Finished the conversion." << std::endl;

  return 0;
}