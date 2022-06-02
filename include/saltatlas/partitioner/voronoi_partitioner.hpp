// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>

namespace saltatlas {

template <typename DistType, typename Point>
class voronoi_partitioner {
 public:
  voronoi_partitioner(ygm::comm &c, hnswlib::SpaceInterface<DistType> &space)
      : m_comm(c), m_space(space) {}

  void initialize(ygm::container::bag<std::pair<uint64_t, Point>> &data,
                  const uint32_t num_partitions) {
    uint64_t num_points = data.size();

    std::vector<uint64_t> seed_ids(num_partitions);
    std::vector<Point>    seed_features;
    if (m_comm.rank0()) {
      select_random_seed_ids(num_partitions, num_points, seed_ids);
    }
    MPI_Bcast(seed_ids.data(), num_partitions, MPI_UNSIGNED_LONG_LONG, 0,
              MPI_COMM_WORLD);

    seed_features.resize(num_partitions);
    auto seed_features_ptr = m_comm.make_ygm_ptr(seed_features);

    data.for_all([&seed_ids, &seed_features, &seed_features_ptr,
                  this](const auto &id_point_pair) {
      const auto &[id, point] = id_point_pair;

      auto lower_iter = std::lower_bound(seed_ids.begin(), seed_ids.end(), id);

      if ((lower_iter != seed_ids.end()) && (*lower_iter == id)) {
        auto seed_index = std::distance(seed_ids.begin(), lower_iter);

        this->m_comm.async_bcast(
            [](auto seed_index, const Point &seed, auto seeds_vector_ptr) {
              (*seeds_vector_ptr)[seed_index] = seed;
            },
            seed_index, point, seed_features_ptr);
      }
    });

    m_comm.barrier();

    set_seeds(seed_features);
    fill_seed_hnsw();
  }

  std::vector<uint64_t> find_point_partitions(const Point   &features,
                                              const uint32_t num_partitions) {
    std::vector<uint64_t> to_return(num_partitions);

    std::priority_queue<std::pair<float, hnswlib::labeltype>> nearest_seeds_pq =
        m_seed_hnsw_ptr->searchKnn(&features, num_partitions);

    uint32_t i = num_partitions;
    while (nearest_seeds_pq.size() > 0) {
      auto seed_ID   = nearest_seeds_pq.top().second;
      to_return[--i] = seed_ID;
      nearest_seeds_pq.pop();
    }

    return to_return;
  }

  void fill_seed_hnsw() {
    m_seed_hnsw_ptr = std::make_unique<hnswlib::HierarchicalNSW<DistType>>(
        &m_space, m_seeds.size(), 16, 200, 3149);

#pragma omp parallel for
    for (uint32_t i = 0; i < m_seeds.size(); ++i) {
      m_seed_hnsw_ptr->addPoint(&m_seeds[i], i);
    }
  }

  void set_seeds(const std::vector<Point> &seed_features) {
    m_seeds.clear();

    for (uint32_t i = 0; i < seed_features.size(); ++i) {
      m_seeds.push_back(seed_features[i]);
    }
  }

  void select_random_seed_ids(const uint32_t         num_seeds,
                              const uint64_t         num_points,
                              std::vector<uint64_t> &seed_ids) {
    std::mt19937                          rng(123);
    std::uniform_int_distribution<size_t> uni(0, num_points - 1);

    for (uint32_t i = 0; i < num_seeds; ++i) {
      seed_ids[i] = uni(rng);
    }

    std::sort(seed_ids.begin(), seed_ids.end());
    seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()),
                   seed_ids.end());

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

 private:
  ygm::comm &m_comm;

  hnswlib::SpaceInterface<DistType> &m_space;

  std::unique_ptr<hnswlib::HierarchicalNSW<DistType>> m_seed_hnsw_ptr;
  std::vector<Point>                                  m_seeds;
};

}  // namespace saltatlas
