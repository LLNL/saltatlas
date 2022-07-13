// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <limits>
#include <random>
#include <ygm/container/bag.hpp>

#if __has_include(<saltatlas_h5_io/h5_reader.hpp>)
#include <saltatlas_h5_io/h5_reader.hpp>
#endif

namespace saltatlas {
namespace dhnsw_detail {

void select_random_seed_ids(const int num_seeds, const size_t num_points,
                            std::vector<size_t> &seed_ids) {
  std::mt19937                          rng(123);
  std::uniform_int_distribution<size_t> uni(0, num_points - 1);

  for (int i = 0; i < num_seeds; ++i) {
    seed_ids[i] = uni(rng);
  }

  std::sort(seed_ids.begin(), seed_ids.end());
  seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()), seed_ids.end());

  while (seed_ids.size() < num_seeds) {
    // std::cout << "Replacing duplicate seed_ids" << std::endl;
    int s = seed_ids.size();
    for (int i = 0; i < num_seeds - s; ++i) {
      seed_ids.push_back(uni(rng));
    }
    std::sort(seed_ids.begin(), seed_ids.end());
    seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()),
                   seed_ids.end());
  }

  assert(seed_ids.size() == num_seeds);

  return;
}

// only include if using HDF5 io helpers
#if __has_include(<saltatlas_h5_io/h5_reader.hpp>)
int get_num_columns(ygm::container::bag<std::string> &bag_filenames,
                    ygm::comm                        &comm) {
  int min_cols{std::numeric_limits<int>::max()};
  int max_cols{-1};

  auto count_columns_lambda = [&min_cols, &max_cols](const auto &fname) {
    saltatlas::h5_io::h5_reader reader(fname);

    if (!reader.is_open()) {
      std::cerr << "Failed to open " << fname << std::endl;
      exit(EXIT_FAILURE);
    }

    int cols = reader.column_names().size();
    min_cols = std::min(min_cols, cols);
    max_cols = std::max(max_cols, cols);
  };
  bag_filenames.for_all(count_columns_lambda);

  bag_filenames.comm().barrier();

  min_cols = comm.all_reduce_min(min_cols);
  max_cols = comm.all_reduce_max(max_cols);

  if (comm.rank0()) {
    if (max_cols != min_cols) {
      std::cerr << "Number of columns inconsistent across files" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  return max_cols;
}

void fill_seed_vector_from_hdf5(const std::vector<size_t>      &seed_ids,
                                const std::vector<std::string> &hdf_file_paths,
                                const std::vector<size_t>      &index_offsets,
                                auto seed_features_ptr, auto &comm) {
  auto store_seed_features_lambda =
      [](size_t seed_index, std::vector<double> vals, auto seeds_vector_ptr) {
        (*seeds_vector_ptr)[seed_index] = vals;
      };

  int curr_offset_ptr = 0;
  // TODO: Make sure < and >= are right inequalities here
  for (int i = 0; i < seed_ids.size();) {
    while (index_offsets[curr_offset_ptr + 1] < seed_ids[i]) {
      ++curr_offset_ptr;
    }

    // Am I responsible for this file?
    if (curr_offset_ptr % comm.size() == comm.rank()) {
      saltatlas::h5_io::h5_reader reader(hdf_file_paths[curr_offset_ptr]);

      if (!reader.is_open()) {
        std::cerr << "Failed to open " << hdf_file_paths[curr_offset_ptr]
                  << std::endl;
        exit(EXIT_FAILURE);
      }

      std::vector<std::string> cols = {"col001", "col040", "col048", "col055"};
      const auto data = reader.read_columns_row_wise<double>(cols);

      while (index_offsets[curr_offset_ptr + 1] >= seed_ids[i] &&
             i < seed_ids.size()) {
        auto seed_features = data[seed_ids[i] - index_offsets[curr_offset_ptr]];
        for (int dest = 0; dest < comm.size(); ++dest) {
          comm.async(dest, store_seed_features_lambda, i, seed_features,
                     seed_features_ptr);
        }
        ++i;
      }
    } else {
      while (index_offsets[curr_offset_ptr + 1] >= seed_ids[i] &&
             i < seed_ids.size()) {
        ++i;
      }
    }
  }
  comm.barrier();

  return;
}

void fill_seed_vector_from_hdf5(const std::vector<size_t>        &seed_ids,
                                ygm::container::bag<std::string> &bag_filenames,
                                const std::vector<std::string>   &col_names,
                                auto &seed_features, auto &comm) {
  seed_features.resize(seed_ids.size());
  // Make ygm pointer to seed_features vector
  auto seed_features_ptr = comm.make_ygm_ptr(seed_features);

  auto store_seed_features_lambda =
      [](size_t seed_index, std::vector<float> vals, auto seeds_vector_ptr) {
        (*seeds_vector_ptr)[seed_index] = vals;
      };

  auto fill_features_lambda = [&store_seed_features_lambda, &seed_ids, &comm,
                               &seed_features_ptr,
                               &col_names](const auto &fname) {
    saltatlas::h5_io::h5_reader reader(fname);

    if (!reader.is_open()) {
      std::cerr << "Failed to open " << fname << std::endl;
      exit(EXIT_FAILURE);
    }

    const auto data_indices = reader.read_column<uint64_t>("index");
    const auto data         = reader.read_columns_row_wise<float>(col_names);

    for (size_t i = 0; i < data_indices.size(); ++i) {
      auto lower_iter =
          std::lower_bound(seed_ids.begin(), seed_ids.end(), data_indices[i]);
      if ((lower_iter != seed_ids.end()) && (*lower_iter == data_indices[i])) {
        size_t seed_index    = std::distance(seed_ids.begin(), lower_iter);
        auto   seed_features = data[i];
        for (int dest = 0; dest < comm.size(); ++dest) {
          comm.async(dest, store_seed_features_lambda, seed_index,
                     seed_features, seed_features_ptr);
        }
      }
    }
  };
  bag_filenames.for_all(fill_features_lambda);

  comm.barrier();

  return;
}

uint64_t count_points_hdf5(ygm::container::bag<std::string> &bag_filenames) {
  size_t num_points{0};

  auto count_points_lambda = [&num_points](const auto &fname) {
    saltatlas::h5_io::h5_reader reader(fname);

    if (!reader.is_open()) {
      std::cerr << "Failed to open " << fname << std::endl;
      exit(EXIT_FAILURE);
    }

    const auto data_indices = reader.read_column<uint64_t>("index");
    num_points += data_indices.size();
  };

  bag_filenames.for_all(count_points_lambda);
  num_points = bag_filenames.comm().all_reduce_sum(num_points);

  return num_points;
}
#endif

}  // namespace dhnsw_detail
}  // namespace saltatlas
