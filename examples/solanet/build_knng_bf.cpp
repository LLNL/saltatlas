// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>

#include "saltatlas/common/detail/neighbor.hpp"
#include "saltatlas/dnnd/detail/knn_heap.hpp"
#include "saltatlas/dnnd/detail/utilities/file.hpp"
#include "saltatlas/dnnd/detail/utilities/omp.hpp"
#include "saltatlas/dnnd/distance.hpp"

#include "shm_query/packed_point_store.hpp"

using namespace saltatlas;
namespace md = metall::mtlldetail;

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using distance_type =
    std::conditional_t<std::is_same_v<fe_type, double>, double, float>;

using point_store_type = saltatlas::packed_point_store<id_type, fe_type>;
using neighbor_type    = detail::neighbor<id_type, distance_type>;
using knng_type =
    boost::unordered::unordered_flat_map<id_type, std::vector<neighbor_type>>;
using knn_heap = dndetail::unique_knn_heap<id_type, distance_type>;

// Used for distance computation
using point_wrapper_type = std::span<fe_type>;

template <typename knng_type>
void dump_knng(const std::filesystem::path& knng_out_path,
               const knng_type& knng, bool dump_distance = false);

struct option {
  std::filesystem::path point_files_dir;
  std::string           point_file_format;
  int                   k{0};
  std::filesystem::path knng_out_dir;
  std::string           distance_func_name;
  bool                  dump_distance{false};
};

// parse CLI arguments
bool parse_options(int argc, char* argv[], option& opt) {
  opt = option();

  int c;
  while ((c = getopt(argc, argv, "i:k:p:f:o:D")) != -1) {
    switch (c) {
      case 'i':
        opt.point_files_dir = optarg;
        break;

      case 'p':
        opt.point_file_format = optarg;
        break;

      case 'k':
        opt.k = std::stoi(optarg);
        break;

      case 'o':
        opt.knng_out_dir = optarg;
        break;

      case 'f':
        opt.distance_func_name = optarg;
        break;

      case 'D':
        opt.dump_distance = true;
        break;

      default:
        std::cerr << "Invalid option" << std::endl;
        return false;
    }
  }

  if (opt.point_files_dir.empty() || opt.k <= 0 || opt.knng_out_dir.empty() ||
      opt.distance_func_name.empty()) {
    std::cerr << "Invalid options" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char* argv[]) {
  option opt;
  if (!parse_options(argc, argv, opt)) {
    return 1;
  }

  std::cout << "Load point" << std::endl;
  const auto point_file_paths =
      saltatlas::dndetail::find_file_paths(opt.point_files_dir);
  const auto points = point_store_type(point_file_paths, opt.point_file_format);

  std::cout << "\nInitialize kNNG" << std::endl;
  knng_type knng;
  for (id_type pid = 0; pid < points.num_points(); ++pid) {
    knng[pid];
  }

  std::cout << "\nConstructing kNNG..." << std::endl;
  auto distance_func =
      saltatlas::distance::distance_function<point_wrapper_type, distance_type>(
          opt.distance_func_name.c_str());
  const auto dims = points.num_dimensions();
  // Brute-force kNNG construction
  OMP_DIRECTIVE(parallel for)
  for (id_type pid = 0; pid < points.num_points(); ++pid) {
    knn_heap neighbors(opt.k);
    for (id_type nid = 0; nid < points.num_points(); ++nid) {
      if (pid == nid) {
        continue;
      }
      const auto distance =
          distance_func(std::span(const_cast<fe_type*>(points.at(pid)), dims),
                        std::span(const_cast<fe_type*>(points.at(nid)), dims));
      neighbors.try_add(nid, distance);
    }
    while (!neighbors.empty()) {
      const auto neighbor = neighbors.top();
      knng[pid].emplace_back(neighbor.id, neighbor.distance);
      neighbors.pop();
    }
    std::sort(knng[pid].begin(), knng[pid].end());
  }

  std::cout << "\nDump kNNG" << std::endl;
  dump_knng(opt.knng_out_dir, knng, opt.dump_distance);

  std::cout << "\nFinished" << std::endl;
  return 0;
}

template <typename knng_type>
void dump_knng(const std::filesystem::path& knng_out_path,
               const knng_type& knng, bool dump_distance) {
  std::ofstream ofs(knng_out_path);

  if (!ofs.is_open()) {
    std::cerr << "Failed to create kNNG file" << std::endl;
    return;
  }

  for (const auto& elem : knng) {
    ofs << elem.first;
    for (const auto& neighbor : elem.second) {
      ofs << " " << neighbor.id;
    }
    ofs << "\n";

    if (!dump_distance) continue;
    ofs << "0.0";  // dummy
    for (const auto& neighbor : elem.second) {
      ofs << " " << neighbor.distance;
    }
    ofs << "\n";
  }
  ofs.close();
}