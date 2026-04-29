// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>

using id_type = uint32_t;
#ifdef SALTATLAS_FEATURE_ELEMENT_TYPE
using fe_type = SALTATLAS_FEATURE_ELEMENT_TYPE;
#else
using fe_type = float;
#endif
using dist_type = float;
using e2i_id_map_type = std::unordered_map<id_type, id_type>;

inline void show_index_score(const dist_type* dists, const size_t n_points,
                             const size_t k, const bool optimized) {
  if (!dists || n_points == 0 || k == 0) {
    std::cout << "Index average neighbor distance: N/A" << std::endl;
    return;
  }
  double       total_distance = 0.0;
  size_t       total_edges    = 0;
  const size_t total          = n_points * k;
  for (size_t i = 0; i < total; ++i) {
    total_distance += static_cast<double>(dists[i]);
  }
  total_edges      = total;
  const double avg = total_distance / static_cast<double>(total_edges);
  if (optimized) {
    std::cout << "Index average neighbor distance (pre-opt): " << avg
              << std::endl;
  } else {
    std::cout << "Index average neighbor distance: " << avg << std::endl;
  }
}

inline void dump_knng(
    const saltatlas::solanet::apu_nn::matrix_view<id_type>&   knn_ids,
    const saltatlas::solanet::apu_nn::matrix_view<dist_type>& knn_dists,
    const std::filesystem::path&                              output_path,
    const bool dump_distance = false) {
  std::ofstream ofs(output_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open output file: " +
                             output_path.string());
  }
  const size_t n_points = knn_ids.n_rows();
  const size_t k        = knn_ids.n_cols();
  for (size_t sid = 0; sid < n_points; ++sid) {
    ofs << sid << " ";
    for (size_t i = 0; i < k; ++i) {
      const id_type nid = knn_ids(sid, i);
      ofs << nid;
      if (i + 1 < k) {
        ofs << " ";
      }
    }
    ofs << "\n";
    if (dump_distance) {
      ofs << "0.0 ";
      for (size_t i = 0; i < k; ++i) {
        const dist_type dist = knn_dists(sid, i);
        ofs << dist;
        if (i + 1 < k) {
          ofs << " ";
        } else {
          ofs << "\n";
        }
      }
    }
  }
  ofs.close();
}

inline void print_time_table() {
  if (saltatlas::rec_time().num_running_timers() > 0) {
    spdlog::error("Some timers are still running.");
  } else {
    std::cout << "\n====================" << std::endl;
    std::cout << "Time table (seconds):" << std::endl;
    std::cout << "====================" << std::endl;
    const auto& time_table = saltatlas::rec_time().get_time_table();
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& entry : time_table) {
      for (std::size_t i = 0; i < entry.depth; ++i) {
        std::cout << "  ";
      }
      std::cout << entry.name << ":\t" << entry.t << std::endl;
    }
  }
}