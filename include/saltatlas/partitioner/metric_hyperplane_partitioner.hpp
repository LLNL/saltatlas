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
class metric_hyperplane_partitioner {
 private:
  struct tree_node {
    DistType                theta;
    std::pair<Point, Point> selectors;
  };

 public:
  metric_hyperplane_partitioner(ygm::comm                         &c,
                                hnswlib::SpaceInterface<DistType> &space)
      : m_comm(c), m_space(space) {}

  void initialize(ygm::container::bag<std::pair<uint64_t, Point>> &data,
                  const uint32_t num_partitions) {
    m_hnsw_ptr = std::make_unique<hnswlib::HierarchicalNSW<DistType>>(
        &m_space, num_partitions, 16, 200, 3149);

    m_num_partitions = num_partitions;
    m_num_levels     = log2(num_partitions);

    std::vector<std::vector<Point>> current_level_points(1);
    std::vector<std::vector<Point>> next_level_points;

    // Stores assignments of points to tree nodes, indexed globally (to match
    // m_tree nodes)
    std::unordered_map<uint64_t, uint64_t> point_assignments;

    data.for_all(
        [&current_level_points, &point_assignments](const auto &id_point) {
          const auto &[id, point] = id_point;
          current_level_points[0].push_back(point);
          point_assignments[id] = 0;
        });

    m_tree.resize((1 << m_num_levels) - 1);

    for (uint32_t l = 0; l < m_num_levels; ++l) {
      m_comm.cout0("Partitioner level: ", l);

      uint32_t num_level_nodes = ((uint32_t)1) << l;
      next_level_points.resize(2 * num_level_nodes);

      for (uint32_t node = 0; node < num_level_nodes; ++node) {
        uint32_t index = ln_to_index(l, node);

        std::pair<Point, Point> selectors =
            choose_selectors(current_level_points[node]);

        m_tree[index].selectors = selectors;

        if (l == m_num_levels - 1) {
          m_hnsw_ptr->addPoint(&m_tree[index].selectors.first, 2 * node);
          m_hnsw_ptr->addPoint(&m_tree[index].selectors.second, 2 * node + 1);
        }
      }

      auto node_thetas =
          calculate_thetas(num_level_nodes, point_assignments, data);

      find_split_thetas(node_thetas, l);

      assign_points(point_assignments, next_level_points, data);

      current_level_points.clear();
      std::swap(current_level_points, next_level_points);
    }
  }

  std::vector<uint64_t> find_point_partitions(const Point   &features,
                                              const uint32_t num_partitions) {
    std::vector<uint64_t> to_return;
    to_return.reserve(num_partitions);

    to_return.push_back(search_tree(features));

    std::priority_queue<std::pair<DistType, hnswlib::labeltype>> nearest_pq =
        m_hnsw_ptr->searchKnn(&features, num_partitions);

    /*
// Need to read into tmp_vec because nearest_pq is in reverse order...
std::vector<uint64_t> tmp_vec(num_partitions);
uint32_t              i = num_partitions;
while (nearest_pq.size() > 0) {
auto seed_ID = nearest_pq.top().second;
tmp_vec[--i] = seed_ID;
nearest_pq.pop();
}
    */

    auto hnsw_nearest =
        m_hnsw_ptr->searchKnnCloserFirst(&features, num_partitions);

    size_t i = 0;
    while (to_return.size() < num_partitions) {
      auto seed_ID = hnsw_nearest[i].second;
      if (seed_ID != to_return[0]) {
        to_return.push_back(seed_ID);
      }
      ++i;
    }

    return to_return;
  }

 private:
  uint32_t log2(uint32_t a) {
    uint32_t to_return = 0;
    while (a >>= 1) {
      ++to_return;
    }

    return to_return;
  }

  uint32_t ln_to_index(const uint32_t level, const uint32_t node) {
    return (((uint32_t)1) << level) - 1 + node;
  }

  std::pair<uint32_t, uint32_t> index_to_ln(const uint32_t index) {
    uint32_t tmp = index;

    uint32_t level = log2(tmp + 1);
    uint32_t node  = index - (((uint32_t)1) << level) + 1;

    return std::make_pair(level, node);
  }

  // Done poorly for prototyping...
  DistType median(const std::vector<DistType> &vals, const int rank) {
    std::vector<DistType> tmp_vals;
    auto                  tmp_ptr = m_comm.make_ygm_ptr(tmp_vals);

    m_comm.async(
        rank,
        [](const auto &vals, auto ptr) {
          for (const auto &val : vals) {
            ptr->push_back(val);
          }
        },
        vals, tmp_ptr);

    m_comm.barrier();

    DistType to_return;
    auto     to_return_ptr = m_comm.make_ygm_ptr(to_return);

    if (m_comm.rank() == rank) {
      std::sort(tmp_vals.begin(), tmp_vals.end());
      to_return = tmp_vals[tmp_vals.size() / 2];

      m_comm.async_bcast(
          [](const auto to_return, auto to_return_ptr) {
            (*to_return_ptr) = to_return;
          },
          to_return, to_return_ptr);
    }

    m_comm.barrier();

    return to_return;
  }

  uint64_t search_tree(const Point &point) {
    uint64_t tree_index = 0;
    while (tree_index < m_tree.size()) {
      const auto &node = m_tree[tree_index];

      DistType dist1 = m_space.get_dist_func()(&point, &node.selectors.first,
                                               m_space.get_dist_func_param());
      DistType dist2 = m_space.get_dist_func()(&point, &node.selectors.second,
                                               m_space.get_dist_func_param());

      DistType theta = pow(dist2, 2) - pow(dist1, 2);

      if (theta < node.theta) {
        tree_index = tree_index * 2 + 1;
      } else {
        tree_index = tree_index * 2 + 2;
      }
    }

    return index_to_ln(tree_index).second;
  }

  // Selectors are not currently chosen uniformly
  std::pair<Point, Point> choose_selectors(
      const std::vector<Point> &node_points) {
    std::pair<Point, Point> to_return;
    auto                    to_return_ptr = m_comm.make_ygm_ptr(to_return);

    std::default_random_engine            gen;
    std::uniform_real_distribution<float> dist(0.0, 1000.0);

    // Choose first point

    std::pair<float, int> sample_selector;
    if (node_points.size() > 0) {
      sample_selector = std::make_pair(dist(gen), m_comm.rank());
    } else {
      sample_selector = std::make_pair(-1.0, m_comm.rank());
    }

    std::pair<float, int> max = m_comm.all_reduce(
        sample_selector,
        [](const auto &a, const auto &b) { return a > b ? a : b; });

    bool     sampled_from{false};
    uint32_t sampled_index;

    if (max.second == m_comm.rank()) {
      std::uniform_int_distribution<uint32_t> array_dist(
          0, node_points.size() - 1);

      sampled_from  = true;
      sampled_index = array_dist(gen);

      m_comm.async_bcast(
          [](const auto &sampled_point, auto to_return_ptr) {
            (*to_return_ptr).first = sampled_point;
          },
          node_points[sampled_index], to_return_ptr);
    }

    // Choose second point
    sample_selector;
    if ((node_points.size() == 0) ||
        (node_points.size() == 1 && sampled_from)) {
      sample_selector = std::make_pair(-1.0, m_comm.rank());
    } else {
      sample_selector = std::make_pair(dist(gen), m_comm.rank());
    }

    max = m_comm.all_reduce(sample_selector, [](const auto &a, const auto &b) {
      return a > b ? a : b;
    });

    if (max.second == m_comm.rank()) {
      std::uniform_int_distribution<uint32_t> array_dist(
          0, node_points.size() - 1);

      auto index = array_dist(gen);
      while (sampled_from && index == sampled_index) {
        index = array_dist(gen);
      }

      m_comm.async_bcast(
          [](const auto &sampled_point, auto to_return_ptr) {
            (*to_return_ptr).second = sampled_point;
          },
          node_points[index], to_return_ptr);
    }

    m_comm.barrier();

    return to_return;
  }

  // TODO: make point_assignments const
  std::vector<std::vector<DistType>> calculate_thetas(
      uint32_t                                         num_nodes,
      std::unordered_map<uint64_t, uint64_t>          &point_assignments,
      ygm::container::bag<std::pair<uint64_t, Point>> &data) {
    std::vector<std::vector<DistType>> thetas(num_nodes);

    data.for_all(
        [&point_assignments, &thetas, this](const auto index_point_pair) {
          const auto &[index, point] = index_point_pair;

          const auto tree_index = point_assignments[index];
          auto      &node       = this->m_tree[tree_index];

          DistType dist1 = m_space.get_dist_func()(
              &point, &node.selectors.first, m_space.get_dist_func_param());
          DistType dist2 = m_space.get_dist_func()(
              &point, &node.selectors.second, m_space.get_dist_func_param());

          DistType theta = pow(dist2, 2) - pow(dist1, 2);

          ASSERT_RELEASE(index_to_ln(point_assignments[index]).second <
                         thetas.size());

          thetas[index_to_ln(point_assignments[index]).second].push_back(theta);
        });

    return thetas;
  }

  void find_split_thetas(const std::vector<std::vector<DistType>> &thetas,
                         const uint32_t                            level) {
    for (int i = 0; i < thetas.size(); ++i) {
      auto theta_median = median(thetas[i], i % m_comm.size());

      auto index = ln_to_index(level, i);

      m_tree[index].theta = theta_median;
    }
  }

  void assign_points(std::unordered_map<uint64_t, uint64_t> &point_assignments,
                     std::vector<std::vector<Point>>        &next_level_points,
                     ygm::container::bag<std::pair<uint64_t, Point>> &data) {
    data.for_all([&point_assignments, &next_level_points,
                  this](const auto &index_point_pair) {
      const auto &[index, point] = index_point_pair;

      const auto tree_index = point_assignments[index];
      auto      &node       = this->m_tree[tree_index];

      DistType dist1 = m_space.get_dist_func()(&point, &node.selectors.first,
                                               m_space.get_dist_func_param());
      DistType dist2 = m_space.get_dist_func()(&point, &node.selectors.second,
                                               m_space.get_dist_func_param());

      DistType theta = pow(dist2, 2) - pow(dist1, 2);

      if (theta < node.theta) {
        point_assignments[index] = 2 * tree_index + 1;

        auto [level, node] = index_to_ln(2 * tree_index + 1);
        next_level_points[node].push_back(point);
      } else {
        point_assignments[index] = 2 * tree_index + 2;

        auto [level, node] = index_to_ln(2 * tree_index + 2);
        next_level_points[node].push_back(point);
      }
    });
  }

  ygm::comm &m_comm;

  hnswlib::SpaceInterface<DistType> &m_space;

  std::vector<tree_node> m_tree;

  uint32_t m_num_partitions;
  uint32_t m_num_levels;
  uint32_t m_max_nonleaf;

  std::unique_ptr<hnswlib::HierarchicalNSW<DistType>> m_hnsw_ptr;
};

}  // namespace saltatlas
