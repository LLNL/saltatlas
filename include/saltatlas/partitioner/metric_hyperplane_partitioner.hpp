// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility.hpp>

namespace saltatlas {

template <typename DistType, typename IndexType, typename Point>
class metric_hyperplane_partitioner {
 public:
  using dist_type  = DistType;
  using index_type = IndexType;
  using point_type = Point;

 private:
  struct tree_node {
    dist_type                      theta;
    std::pair<point_type, point_type> selectors;
  };

 public:
  metric_hyperplane_partitioner(ygm::comm                       &c,
                                hnswlib::SpaceInterface<dist_type> &space)
      : m_comm(c), m_space(space) {}

  ~metric_hyperplane_partitioner() {
    m_comm.cout0("Median time: ", m_median_time);
    m_comm.barrier();
  }

  template <class Container>
  void initialize(Container &data, const uint32_t num_partitions) {
    m_hnsw_ptr = std::make_unique<hnswlib::HierarchicalNSW<dist_type>>(
        &m_space, num_partitions, 16, 200, 3149);

    m_num_partitions   = num_partitions;
    m_num_levels       = log2(num_partitions) + 1;
    m_first_leaf_index = (((uint32_t)1) << (m_num_levels - 1)) - 1;

    std::vector<std::vector<point_type>> current_level_points(1);
    std::vector<std::vector<point_type>> next_level_points;

    // Stores assignments of points to tree nodes, indexed globally (to match
    // m_tree nodes)
    std::unordered_map<index_type, index_type> point_assignments;

    ygm::timer t{};

    data.for_all([&current_level_points, &point_assignments](
                     const auto &id, const auto &point) {
      current_level_points[0].push_back(point);
      point_assignments[id] = 0;
    });

    m_tree.resize((1 << m_num_levels - 1) - 1);

    for (uint32_t l = 0; l < m_num_levels - 1; ++l) {
      uint32_t num_level_nodes = ((uint32_t)1) << l;
      next_level_points.resize(2 * num_level_nodes);

      std::vector<std::pair<point_t, point_t>> best_selectors(num_level_nodes);
      std::vector<dist_t>                      best_thetas(num_level_nodes,
                                                           std::numeric_limits<dist_t>::max());

      for (int selector_trial = 0; selector_trial < max_selector_trials;
           ++selector_trial) {
        std::vector<std::pair<point_t, point_t>> selector_pairs;
        for (uint32_t node = 0; node < num_level_nodes; ++node) {
          selector_pairs.push_back(choose_selectors(
              current_level_points[node],
              l * max_selector_trials * num_level_nodes +
                  selector_trial * num_level_nodes + node + 654321));
        }

        auto node_thetas = calculate_thetas(num_level_nodes, point_assignments,
                                            data, selector_pairs);

        for (uint32_t node = 0; node < num_level_nodes; ++node) {
          uint32_t index = ln_to_index(l, node);

          std::pair<point_type, point_type> theta_median =
              sampled_median(node_thetas[node], 0.01,
                             l * max_selector_trials * num_level_nodes +
                                 selector_trial * num_level_nodes + node);

          if (std::abs(theta_median) < std::abs(best_thetas[node])) {
            best_thetas[node]    = theta_median;
            best_selectors[node] = selector_pairs[node];
          }
        }
      }

      // Record best thetas and associated selectors
      for (uint32_t node = 0; node < num_level_nodes; ++node) {
        uint32_t index = ln_to_index(l, node);

        m_tree[index].selectors = best_selectors[node];
        m_tree[index].theta     = best_thetas[node];
      }

      assign_points(point_assignments, next_level_points, data);

      current_level_points.clear();
      std::swap(current_level_points, next_level_points);
    }

    m_comm.cout0("Adding points to HNSW");
    // Add bottom level to hnsw
    // Bottom level only exists implicitly as the split points to the
    // second-to-last layer
    for (uint32_t i = 0; i < (((uint32_t)1) << m_num_levels - 2); ++i) {
      auto tree_index = ln_to_index(m_num_levels - 2, i);
      m_hnsw_ptr->addPoint(&m_tree[tree_index].selectors.first, 2 * i);
      m_hnsw_ptr->addPoint(&m_tree[tree_index].selectors.second, 2 * i + 1);
    }

    m_comm.cout0("Partitioner initialization time: ", t.elapsed());
  }

  std::vector<index_type> find_point_partitions(
      const point_type &features, const uint32_t num_partitions) const {
    std::vector<index_type> to_return;
    to_return.reserve(num_partitions);

    auto search_tree_results = search_tree(features);
    ASSERT_RELEASE(search_tree_results < m_num_partitions);

    to_return.push_back(search_tree_results);

    auto hnsw_nearest =
        m_hnsw_ptr->searchKnnCloserFirst(&features, num_partitions);

    size_t i = 0;
    while (to_return.size() < num_partitions) {
      ASSERT_RELEASE(i < hnsw_nearest.size());
      index_type seed_ID = hnsw_nearest[i].second;
      ASSERT_RELEASE(seed_ID < m_num_partitions);
      if (seed_ID != to_return[0]) {
        to_return.push_back(seed_ID);
      }
      ++i;
    }

    return to_return;
  }

  std::vector<point_type> find_point_partition_representatives(
      const point_type &features, const uint32_t num_partitions) const {
    auto partition_index_vec = find_point_partitions(features, num_partitions);
    std::vector<point_type> to_return;
    to_return.reserve(num_partitions);

    for (const auto partition_index : partition_index_vec) {
      to_return.push_back(get_partition_representative(partition_index));
    }

    return to_return;
  }

  point_type find_point_partition_representative(const point_type &features) const {
    auto partition_rep_vec = find_point_partition_representatives(features, 1);
    return partition_rep_vec[0];
  }

  uint32_t num_partitions() const { return m_num_partitions; }

  std::vector<dist_type> get_thetas() const {
    std::vector<dist_type> to_return;

    for (const auto &node : m_tree) {
      dist_type seed_dist =
          m_space.get_dist_func()(&node.selectors.first, &node.selectors.second,
                                  m_space.get_dist_func_param());
      to_return.push_back(node.theta / pow(seed_dist, 2));
      // to_return.push_back(node.theta);
    }

    return to_return;
  }

  struct node_statistics {
    std::vector<dist_type>         thetas;
    std::pair<point_type, point_type> selectors;

    dist_type selector_distance;

    index_type              index;
    index_type              parent;
    std::vector<index_type> children;
  };

  template <typename Container>
  std::vector<node_statistics> find_tree_statistics(Container &data) {
    ygm::container::map<index_type, node_statistics> stats_map;

    data.for_all([&stats_map, this](const auto &index, const auto &point) {
      auto leaf_index = search_tree(point);

      auto search_path = reconstruct_search_path(leaf_index);

      for (size_t i = 1; i < search_path.size(); ++i) {
        const auto tree_index = search_path[i];

        auto &node = m_tree[tree_index];

        dist_type dist1 = m_space.get_dist_func()(&point, &node.selectors.first,
                                               m_space.get_dist_func_param());
        dist_type dist2 = m_space.get_dist_func()(&point, &node.selectors.second,
                                               m_space.get_dist_func_param());

        dist_type theta = pow(dist2, 2) - pow(dist1, 2);
      }
    });
  }

  const point_type &get_partition_representative(const index_type idx) const {
    // Get global tree index for idx in lowest level of tree
    auto leaf_index = ln_to_index(m_num_levels - 1, idx);

    if (leaf_index % 2 == 1) {
      return m_tree[(leaf_index - 1) / 2].selectors.first;
    } else {
      return m_tree[(leaf_index - 1) / 2].selectors.second;
    }
  }

  void print_tree() const {
    std::stringstream ss;

    for (int i = 0; i < m_first_leaf_index; ++i) {
      auto &current_node = m_tree[i];
      if (i > 0) {
        auto &parent_node = m_tree[(i - 1) / 2];
        ss << "Rep: ";
        if (i % 2 == 1) {
          ss << parent_node.selectors.first << "\t";
        } else {
          ss << parent_node.selectors.second << "\t";
        }
      } else {
        ss << "ROOT\t";
      }

      ss << "Children: " << current_node.selectors.first << ", "
         << current_node.selectors.second << "\t";
      ss << "theta: " << current_node.theta;

      ss << std::endl;
    }

    m_comm.cout0(ss.str());
  }

 private:
  uint32_t log2(const uint32_t a) const {
    ASSERT_RELEASE(a > 0);
    uint32_t tmp       = a;
    uint32_t to_return = 0;
    while (tmp >>= 1) {
      ++to_return;
    }

    return to_return;
  }

  uint32_t ln_to_index(const uint32_t level, const uint32_t node) const {
    return (((uint32_t)1) << level) - 1 + node;
  }

  std::pair<uint32_t, uint32_t> index_typeo_ln(const uint32_t index) const {
    uint32_t level = log2(index + 1);
    uint32_t node  = index - (((uint32_t)1) << level) + 1;

    return std::make_pair(level, node);
  }

  // Done poorly for prototyping...
  dist_type median(const std::vector<dist_type> &vals, const int rank) {
    std::vector<dist_type> tmp_vals;
    auto                tmp_ptr = m_comm.make_ygm_ptr(tmp_vals);

    m_comm.async(
        rank,
        [](const auto &vals, auto ptr) {
          for (const auto &val : vals) {
            ptr->push_back(val);
          }
        },
        vals, tmp_ptr);

    m_comm.barrier();

    dist_type to_return;
    auto   to_return_ptr = m_comm.make_ygm_ptr(to_return);

    if (m_comm.rank() == rank) {
      std::sort(tmp_vals.begin(), tmp_vals.end());
      ASSERT_RELEASE(tmp_vals.size() > 0);
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

  dist_t sampled_median(const std::vector<dist_t> &vals,
                        const float                sample_ratio = 0.01,
                        const int                  seed         = 1234) {
    std::vector<dist_t>         samples;
    static std::vector<dist_t> &s_samples = samples;
    std::vector<dist_t>         local_samples;

    auto insert_lambda = [](const auto &vals) {
      s_samples.insert(s_samples.end(), vals.begin(), vals.end());
    };

    std::mt19937_64                       gen(m_comm.rank() ^ seed);
    std::uniform_int_distribution<size_t> dist(0, vals.size() - 1);

    int num_samples = std::min<int>(
        vals.size(), static_cast<int>(std::ceil(sample_ratio * vals.size())));

    for (int i = 0; i < num_samples; ++i) {
      auto index = dist(gen);
      local_samples.push_back(vals[index]);
    }

    m_comm.async(0, insert_lambda, local_samples);

    m_comm.barrier();

    dist_t         to_return;
    static dist_t &s_to_return = to_return;

    if (m_comm.rank() == 0) {
      std::sort(samples.begin(), samples.end());
      ASSERT_RELEASE(samples.size() > 0);

      if (samples.size() % 2 == 1) {
        to_return = samples[samples.size() / 2];
      } else {
        auto upper_index = samples.size() / 2;
        to_return = (samples[upper_index - 1] + samples[upper_index]) / 2;
      }

      m_comm.async_bcast([](const auto median) { s_to_return = median; },
                         to_return);
    }

    m_comm.barrier();

    return to_return;
  }

  uint64_t search_tree(const point_type &point) const {
    uint64_t tree_index = 0;
    while (tree_index < m_tree.size()) {
      const auto &node = m_tree[tree_index];

      dist_type dist1 = m_space.get_dist_func()(&point, &node.selectors.first,
                                             m_space.get_dist_func_param());
      dist_type dist2 = m_space.get_dist_func()(&point, &node.selectors.second,
                                             m_space.get_dist_func_param());

      dist_type theta = pow(dist2, 2) - pow(dist1, 2);

      if (theta > node.theta) {
        tree_index = tree_index * 2 + 1;
      } else {
        tree_index = tree_index * 2 + 2;
      }
    }

    auto level_node = index_typeo_ln(tree_index);

    ASSERT_RELEASE(level_node.first < m_num_levels);
    ASSERT_RELEASE(level_node.second < pow(2, level_node.first));

    return level_node.second;
  }

  std::vector<uint64_t> reconstruct_search_path(const uint64_t &leaf_node) {
    std::vector<uint64_t> to_return;

    uint64_t curr_node = leaf_node;
    to_return.push_back(curr_node);
    while (curr_node > 0) {
      curr_node = (curr_node - 1) / 2;

      to_return.push_back(curr_node);
    }

    return to_return;
  }

  // Selectors are not currently chosen uniformly
  std::pair<point_type, point_type> choose_selectors(
      const std::vector<point_type> &node_points, const uint64_t seed = 12345) {
    std::pair<point_type, point_type> to_return;
    auto                        to_return_ptr = m_comm.make_ygm_ptr(to_return);

    std::default_random_engine            gen(m_comm.rank() ^ seed);
    std::uniform_real_distribution<float> dist;

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
      ASSERT_RELEASE(node_points.size() > 0);
      std::uniform_int_distribution<uint32_t> array_dist(
          0, node_points.size() - 1);

      sampled_from  = true;
      sampled_index = array_dist(gen);
      ASSERT_RELEASE(sampled_index < node_points.size());

      auto selector1 = node_points[sampled_index];
      m_comm.async_bcast(
          [](const auto &sampled_point, auto to_return_ptr) {
            (*to_return_ptr).first = sampled_point;
          },
          selector1, to_return_ptr);
    }

    // Choose second point
    to_return.second = to_return.first;
    // loop until first and second selector are different
    while (to_return.second == to_return.first) {
      if ((node_points.size() == 0) ||
          (node_points.size() == 1 && sampled_from)) {
        sample_selector = std::make_pair(-1.0, m_comm.rank());
      } else {
        sample_selector = std::make_pair(dist(gen), m_comm.rank());
      }

      max = m_comm.all_reduce(
          sample_selector,
          [](const auto &a, const auto &b) { return a > b ? a : b; });

      if (max.second == m_comm.rank()) {
        std::uniform_int_distribution<uint32_t> array_dist(
            0, node_points.size() - 1);

        auto index = array_dist(gen);
        while (sampled_from && index == sampled_index) {
          index = array_dist(gen);
        }

        ASSERT_RELEASE(index < node_points.size());
        m_comm.async_bcast(
            [](const auto &sampled_point, auto to_return_ptr) {
              (*to_return_ptr).second = sampled_point;
            },
            node_points[index], to_return_ptr);
      }

      m_comm.barrier();
    }

    return to_return;
  }

  // TODO: make point_assignments const
  template <typename Container>
  std::vector<std::vector<dist_type>> calculate_thetas(
      uint32_t                              num_nodes,
      std::unordered_map<index_type, index_type> &point_assignments, Container &data,
      const std::vector<std::pair<point_type, point_type>> &selector_pairs) {
    std::vector<std::vector<dist_t>> thetas(num_nodes);

    data.for_all([&point_assignments, &thetas, &selector_pairs, this](
                     const auto &index, const auto &point) {
      const auto tree_index = point_assignments[index];
      auto [level, node]    = index_to_ln(tree_index);

      dist_type dist1 = m_space.get_dist_func()(
          &point, &selector_pairs[node].first, m_space.get_dist_func_param());
      dist_type dist2 = m_space.get_dist_func()(
          &point, &selector_pairs[node].second, m_space.get_dist_func_param());

      dist_type theta = pow(dist2, 2) - pow(dist1, 2);

      ASSERT_RELEASE(index_typeo_ln(point_assignments[index]).second <
                     thetas.size());

      thetas[index_typeo_ln(point_assignments[index]).second].push_back(theta);
    });

    return thetas;
  }

  void find_split_thetas(const std::vector<std::vector<dist_type>> &thetas,
                         const uint32_t                          level) {
    for (int i = 0; i < thetas.size(); ++i) {
      m_comm.barrier();
      ygm::timer t{};
      // auto       theta_median = median(thetas[i], i % m_comm.size());
      auto theta_median = sampled_median(thetas[i], 0.01, i);
      m_median_time += t.elapsed();

      auto index = ln_to_index(level, i);

      ASSERT_RELEASE(index < m_tree.size());
      m_tree[index].theta = theta_median;
    }
  }

  template <typename Container>
  void assign_points(
      std::unordered_map<index_type, index_type> &point_assignments,
      std::vector<std::vector<point_type>>       &next_level_points,
      Container                                  &data) {
    data.for_all([&point_assignments, &next_level_points, this](
                     const auto &index, const auto &point) {
      const auto tree_index = point_assignments[index];
      auto      &node       = this->m_tree[tree_index];

      dist_type dist1 = m_space.get_dist_func()(&point, &node.selectors.first,
                                             m_space.get_dist_func_param());
      dist_type dist2 = m_space.get_dist_func()(&point, &node.selectors.second,
                                             m_space.get_dist_func_param());

      dist_type theta = pow(dist2, 2) - pow(dist1, 2);

      if (theta > node.theta) {
        point_assignments[index] = 2 * tree_index + 1;

        auto [level, node] = index_typeo_ln(2 * tree_index + 1);
        ASSERT_RELEASE(node < next_level_points.size());
        next_level_points[node].push_back(point);
      } else {
        point_assignments[index] = 2 * tree_index + 2;

        auto [level, node] = index_typeo_ln(2 * tree_index + 2);
        ASSERT_RELEASE(node < next_level_points.size());
        next_level_points[node].push_back(point);
      }
    });
  }

  ygm::comm &m_comm;

  hnswlib::SpaceInterface<dist_type> &m_space;

  std::vector<tree_node> m_tree;

  uint32_t m_num_partitions;
  uint32_t m_num_levels;
  uint32_t m_first_leaf_index;

  std::unique_ptr<hnswlib::HierarchicalNSW<dist_type>> m_hnsw_ptr;

  double m_median_time{0.0};

  int max_selector_trials = 100;
};

}  // namespace saltatlas
