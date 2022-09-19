// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <ygm/comm.hpp>

namespace saltatlas {

template <typename DistType, typename IndexType, typename Point>
class voronoi_partitioner {
 public:
  using dist_t  = DistType;
  using index_t = IndexType;
  using point_t = Point;

  voronoi_partitioner(ygm::comm &c, hnswlib::SpaceInterface<dist_t> &space)
      : m_comm(c), m_space(space) {}

  template <template <typename, typename> class Container>
  void initialize(Container<index_t, point_t> &data,
                  const uint32_t               num_partitions) {
    size_t num_points = data.size();

    std::vector<index_t> seed_ids(num_partitions);
    std::vector<point_t> seed_features;
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
            [](auto seed_index, const point_t &seed, auto seeds_vector_ptr) {
              (*seeds_vector_ptr)[seed_index] = seed;
            },
            seed_index, point, seed_features_ptr);
      }
    });

    m_comm.barrier();

    set_seeds(seed_features);
    fill_seed_hnsw();
  }

  std::vector<index_t> find_point_partitions(const point_t &features,
                                             const uint32_t num_partitions) {
    std::vector<index_t> to_return(num_partitions);

    std::priority_queue<std::pair<float, hnswlib::labeltype>> nearest_seeds_pq =
        m_seed_hnsw_ptr->searchKnn(&features, num_partitions);

    uint32_t i = num_partitions;
    while (nearest_seeds_pq.size() > 0) {
      index_t seed_ID = nearest_seeds_pq.top().second;
      to_return[--i]  = seed_ID;
      nearest_seeds_pq.pop();
    }

    return to_return;
  }

  void fill_seed_hnsw() {
    m_seed_hnsw_ptr = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(
        &m_space, m_seeds.size(), 16, 200, 3149);

#pragma omp parallel for
    for (uint32_t i = 0; i < m_seeds.size(); ++i) {
      m_seed_hnsw_ptr->addPoint(&m_seeds[i], i);
    }
  }

  void set_seeds(const std::vector<point_t> &seed_features) {
    m_seeds.clear();

    for (uint32_t i = 0; i < seed_features.size(); ++i) {
      m_seeds.push_back(seed_features[i]);
    }
  }

  void select_random_seed_ids(const uint32_t        num_seeds,
                              const uint64_t        num_points,
                              std::vector<index_t> &seed_ids) {
    std::mt19937                           rng(123);
    std::uniform_int_distribution<index_t> uni(0, num_points - 1);

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

  hnswlib::SpaceInterface<dist_t> &m_space;

  std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> m_seed_hnsw_ptr;
  std::vector<point_t>                              m_seeds;
};

}  // namespace saltatlas
