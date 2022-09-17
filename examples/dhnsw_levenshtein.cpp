// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <saltatlas/dhnsw/detail/utility.hpp>
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/partitioner/metric_hyperplane_partitioner.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/set.hpp>
#include <ygm/utility.hpp>

template <typename String>
size_t levenshtein_distance(const String& s1, const String& s2) {
  const size_t m = s1.size();
  const size_t n = s2.size();

  if (m == 0) {
    return n;
  }
  if (n == 0) {
    return m;
  }

  // Row of matrix for dynamic programming approach
  std::vector<size_t> dist_row(m + 1);
  for (size_t i = 0; i < m + 1; ++i) {
    dist_row[i] = i;
  }

  for (size_t i = 1; i < n + 1; ++i) {
    size_t diag = i - 1;
    size_t next_diag;
    dist_row[0] = i;
    for (size_t j = 1; j < m + 1; ++j) {
      next_diag              = dist_row[j];
      bool substitution_cost = (s1[i - 1] != s2[j - 1]);

      dist_row[j] =
          std::min(1 + dist_row[j],
                   std::min(1 + dist_row[j - 1], substitution_cost + diag));
      diag = next_diag;
    }
  }

  return dist_row[m];
}

template <typename String>
float fuzzy_levenshtein(const String& s1, const String& s2) {
  float fuzz = (std::hash<String>()(s1) ^ std::hash<String>()(s2)) / 4.0;
  fuzz /= std::numeric_limits<size_t>::max();
  return levenshtein_distance(s1, s2) + fuzz;
}

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  int voronoi_rank = 3;
  int num_cells    = 4;

  auto fuzzy_leven_space =
      saltatlas::dhnsw_detail::SpaceWrapper(fuzzy_levenshtein<std::string>);

  std::vector<std::string> seeds{"car", "jump", "frog", "test"};
  std::vector<std::string> strings{"cat",   "dog",  "apple", "desk",
                                   "floor", "lamp", "car",   "flag"};

  using dist_t  = float;
  using index_t = std::size_t;
  using point_t = std::string;

  ygm::container::bag<std::pair<index_t, point_t>> string_bag(world);
  if (world.rank0()) {
    index_t i{0};
    for (const auto& s : strings) {
      string_bag.async_insert(std::make_pair(i++, s));
    }
  }

  saltatlas::metric_hyperplane_partitioner<dist_t, index_t, point_t>
      partitioner(world, fuzzy_leven_space);

  saltatlas::dhnsw dist_index(voronoi_rank, num_cells, &fuzzy_leven_space,
                              &world, partitioner);

  world.barrier();
  ygm::timer t{};

  world.cout0("Partitioning data");
  dist_index.partition_data(string_bag, num_cells);

  world.barrier();

  world.cout0("Distributing data to local HNSWs");
  int index{0};
  if (world.rank0()) {
    for (const auto& string : strings) {
      dist_index.queue_data_point_insertion(index++, string);
    }
  }

  world.barrier();

  world.cout0("Initializing local HNSWs");
  dist_index.initialize_hnsw();

  world.barrier();
  world.cout0("Total build time: ", t.elapsed());

  world.cout0("Global HNSW size: ", dist_index.global_size());

  std::string test_string("flap");
  world.cout0("Attempting query for '", test_string, "'");
  world.barrier();

  auto fuzzy_result_lambda =
      [](const point_t&                        query_string,
         const std::multimap<dist_t, index_t>& nearest_neighbors,
         auto                                  dist_knn_index) {
        for (const auto& result_pair : nearest_neighbors) {
          dist_knn_index->comm().cout()
              << result_pair.second << " fuzzy dist: " << result_pair.first
              << std::endl;
        }
      };
  if (world.rank0()) {
    dist_index.query(test_string, 2, 2, voronoi_rank, 1, fuzzy_result_lambda);
  }

  world.barrier();

  return 0;
}
