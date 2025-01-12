// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <random>
#include <unordered_map>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <saltatlas/common/point_store.hpp>
#include <saltatlas/dhnsw/detail/median.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility.hpp>

namespace saltatlas {
template <typename DistType, typename IndexType, typename Point>
class metric_hyperplane_partitioner {
 public:
  using distance_type     = DistType;
  using id_type           = IndexType;
  using point_type        = Point;
  using tree_node_id_type = int;
  using theta_type        = float;

  using global_tree_node_id_type      = tree_node_id_type;
  using intra_level_tree_node_id_type = tree_node_id_type;
  using tree_level_id_type            = uint16_t;
  /*
  tree_node_id_type as_int(const global_tree_node_id_type id) {
    return static_cast<tree_node_id_type>(id);
  }
  tree_node_id_type as_int(const intra_level_tree_node_id_type id) {
    return static_cast<tree_node_id_type>(id);
  }
  uint16_t as_int(const tree_level_id_type id) {
    return static_cast<uint16_t>(id);
  }
  */

 private:
  struct tree_node_type {
    point_type left_rep;
    point_type right_rep;
    theta_type theta_split = std::numeric_limits<float>::max();
    theta_type rep_dist;
  };

 public:
  metric_hyperplane_partitioner(const uint32_t num_partitions,
                                hnswlib::SpaceInterface<distance_type> &space,
                                ygm::comm                              &c)
      : m_num_partitions(num_partitions), m_space(space), m_comm(c) {
    // Tree is being built without explicitly creating leaves
    // Take log2 to find height of binary tree and fix for incomplete levels
    m_num_levels =
        static_cast<tree_level_id_type>(ceil(log2(m_num_partitions)));

    m_num_tree_nodes =
        static_cast<global_tree_node_id_type>(m_num_partitions - 1);
    m_tree.resize(static_cast<id_type>(m_num_tree_nodes));

    // Initialize RNG
    std::random_device rd;
    m_rng.seed(rd());
  }

  ~metric_hyperplane_partitioner() { m_comm.barrier(); }

  void initialize(point_store<id_type, point_type> &data) {
    std::unordered_map<id_type, global_tree_node_id_type>
        local_point_assignments;
    // For sampling new points for children nodes and updating assignments of
    // points
    std::vector<std::vector<id_type>>  // intra_level_tree_id -> vector of
                                       // points
        local_tree_level_id_splits(1);
    std::vector<tree_node_type>
        candidate_tree_level;  // intra_level_tree_id -> candidate tree nodes
    std::vector<std::vector<theta_type>>
        curr_level_thetas;  // intra_level_tree_id -> vector of theta values

    // Start each iteration with points assigned to nodes within the current
    // level Find representative points to use for splitting data and assign
    // points to nodes in the next level
    for (tree_level_id_type level{0}; level < m_num_levels; ++level) {
      // Find number of tree nodes in current level. Last level may be
      // incomplete.
      intra_level_tree_node_id_type max_nodes_curr_level =
          ((intra_level_tree_node_id_type)1) << level;
      intra_level_tree_node_id_type num_nodes_curr_level =
          std::min<intra_level_tree_node_id_type>(
              max_nodes_curr_level,
              m_num_tree_nodes -
                  ((((intra_level_tree_node_id_type)1) << level) - 1));
      intra_level_tree_node_id_type node_offset_curr_level =
          (((intra_level_tree_node_id_type)1) << level) - 1;

      // Set up the candidate tree level
      candidate_tree_level.clear();
      candidate_tree_level.resize(num_nodes_curr_level);
      local_tree_level_id_splits.clear();
      local_tree_level_id_splits.resize(num_nodes_curr_level);

      assign_points_current_level(level, data, local_tree_level_id_splits,
                                  local_point_assignments);

      // Attempt several trials to find the best splitting points to use
      for (uint32_t trial = 0; trial < m_candidate_level_trials; ++trial) {
        propose_split_points(data, candidate_tree_level,
                             local_tree_level_id_splits);
        auto curr_level_thetas = compute_split_thetas(
            data, local_point_assignments, candidate_tree_level,
            node_offset_curr_level, num_nodes_curr_level);

        auto split_thetas = compute_medians(curr_level_thetas, m_comm);

        for (tree_node_id_type i = 0; i < split_thetas.size(); ++i) {
          tree_node_type &cur_node = m_tree[node_offset_curr_level + i];
          if (std::abs(split_thetas[i]) < std::abs(cur_node.theta_split)) {
            cur_node.theta_split = split_thetas[i];
            cur_node.left_rep    = candidate_tree_level[i].left_rep;
            cur_node.right_rep   = candidate_tree_level[i].right_rep;
            cur_node.rep_dist    = candidate_tree_level[i].rep_dist;
          }
        }
      }
    }

    // Print tree
    if (m_comm.rank0()) {
      for (int i = 0; i < m_tree.size(); ++i) {
        const auto &node = m_tree[i];
      }
    }
  }

  const id_type find_partition(const point_type &point) const {
    tree_node_id_type node_id{0};

    while (node_id < m_num_tree_nodes) {
      theta_type theta = compute_theta(point, m_tree[node_id]);

      std::pair<global_tree_node_id_type, global_tree_node_id_type>
          child_node_ids = get_child_ids(node_id);

      if (theta < m_tree[node_id].theta_split) {
        node_id = child_node_ids.first;
      } else {
        node_id = child_node_ids.second;
      }
    }

    id_type partition = node_id - m_num_tree_nodes;

    assert(partition < m_num_partitions);
    return partition;
  }

  std::vector<std::pair<intra_level_tree_node_id_type,
                        std::reference_wrapper<point_type>>>
  get_leaves() {
    std::vector<std::pair<intra_level_tree_node_id_type,
                          std::reference_wrapper<point_type>>>
        leaves;

    for (global_tree_node_id_type leaf_num = 0; leaf_num < m_num_partitions;
         ++leaf_num) {
      global_tree_node_id_type leaf_node_id   = leaf_num + m_num_tree_nodes;
      global_tree_node_id_type parent_node_id = get_parent_id(leaf_node_id);

      if (is_left_child(leaf_node_id)) {
        leaves.push_back(std::make_pair(
            leaf_num, std::reference_wrapper(m_tree[parent_node_id].left_rep)));
      } else {
        leaves.push_back(std::make_pair(
            leaf_num,
            std::reference_wrapper(m_tree[parent_node_id].right_rep)));
      }
    }

    return leaves;
  }

  uint32_t num_partitions() { return m_num_partitions; }

 private:
  const global_tree_node_id_type level_to_global(
      tree_level_id_type level, intra_level_tree_node_id_type id) const {
    return (((intra_level_tree_node_id_type)1) << level) + (id - 1);
  }

  const std::pair<tree_level_id_type, intra_level_tree_node_id_type>
  global_to_level(const tree_node_id_type global) const {
    tree_level_id_type level = ((tree_level_id_type)floor(log2(global + 1)));
    tree_node_id_type  intra_level_id =
        global - (((global_tree_node_id_type)1) << (level)) + 1;

    return std::make_pair(level, intra_level_id);
  }

  const std::pair<global_tree_node_id_type, global_tree_node_id_type>
  get_child_ids(global_tree_node_id_type parent) const {
    return std::make_pair(2 * parent + 1, 2 * parent + 2);
  }

  const global_tree_node_id_type get_parent_id(
      global_tree_node_id_type child) const {
    return (child - 1) / 2;
  }

  const bool is_left_child(global_tree_node_id_type child) const {
    return (child % 2 == 0);
  }

  const bool is_right_child(global_tree_node_id_type child) const {
    return (child % 2 == 1);
  }

  const theta_type compute_theta(const point_type     &point,
                                 const tree_node_type &node) const {
    distance_type left_dist = m_space.get_dist_func()(
        &point, &(node.left_rep), m_space.get_dist_func_param());
    distance_type right_dist = m_space.get_dist_func()(
        &point, &(node.right_rep), m_space.get_dist_func_param());
    return ((theta_type)pow(left_dist, 2) - pow(right_dist, 2)) /
           pow(node.rep_dist, 2);
  }

  void assign_points_current_level(
      const tree_level_id_type level, point_store<id_type, point_type> &data,
      std::vector<std::vector<id_type>> &local_tree_level_id_splits,
      std::unordered_map<id_type, global_tree_node_id_type>
          &local_point_assignments) {
    if (level == 0) {
      // Assign all items to the root node
      for (auto point_iter = data.begin(); point_iter != data.end();
           ++point_iter) {
        const auto &[id, point]     = *point_iter;
        local_point_assignments[id] = 0;
        local_tree_level_id_splits[0].push_back(id);
      }

    } else {
      // Assign points to the appropriate nodes in the next level
      for (auto point_iter = data.begin(); point_iter != data.end();
           ++point_iter) {
        const auto &[id, point]               = *point_iter;
        global_tree_node_id_type curr_node_id = local_point_assignments[id];
        // Don't assign if point is already at a leaf
        if (curr_node_id < m_num_tree_nodes) {
          tree_node_type curr_node = m_tree[curr_node_id];
          theta_type     theta     = compute_theta(point, curr_node);

          std::pair<global_tree_node_id_type, global_tree_node_id_type>
                                   child_node_ids = get_child_ids(curr_node_id);
          global_tree_node_id_type new_node_id;
          if (theta < curr_node.theta_split) {
            new_node_id = child_node_ids.first;
          } else {
            new_node_id = child_node_ids.second;
          }
          local_point_assignments[id] = new_node_id;

          if (new_node_id < m_num_tree_nodes) {
            intra_level_tree_node_id_type new_intra_level_id =
                global_to_level(new_node_id).second;
            local_tree_level_id_splits[new_intra_level_id].push_back(id);
          }
        }
      }
    }
  }

  void propose_split_points(
      const point_store<id_type, point_type> &data,
      std::vector<tree_node_type>            &candidate_tree_level,
      std::vector<std::vector<id_type>>      &local_tree_level_id_splits) {
    // Both sizes should be the same before looking for split points
    YGM_ASSERT_RELEASE(candidate_tree_level.size() ==
                       local_tree_level_id_splits.size());

    static const std::vector<std::vector<id_type>>
        &s_local_tree_level_id_splits(local_tree_level_id_splits);

    static std::vector<tree_node_type> &s_candidate_tree_level(
        candidate_tree_level);

    static std::vector<size_t> num_local_points_vec;
    static std::vector<size_t> num_global_points_vec;
    static std::vector<size_t> offsets_vec;
    num_local_points_vec.resize(candidate_tree_level.size());
    num_global_points_vec.resize(candidate_tree_level.size());
    offsets_vec.resize(candidate_tree_level.size());

    for (intra_level_tree_node_id_type i = 0;
         i < local_tree_level_id_splits.size(); ++i) {
      num_local_points_vec[i] = local_tree_level_id_splits[i].size();
    }

    MPI_Allreduce(num_local_points_vec.data(), num_global_points_vec.data(),
                  num_local_points_vec.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  m_comm.get_mpi_comm());
    MPI_Exscan(num_local_points_vec.data(), offsets_vec.data(),
               num_local_points_vec.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               m_comm.get_mpi_comm());

    ygm::ygm_ptr<const point_store<id_type, point_type>> p_data(&data);
    p_data.check(m_comm);

    for (intra_level_tree_node_id_type i = 0; i < candidate_tree_level.size();
         ++i) {
      YGM_ASSERT_RELEASE(num_global_points_vec[i] > 1);

      if (i % m_comm.size() == m_comm.rank()) {
        std::uniform_int_distribution<size_t> dist(
            0, num_global_points_vec[i] - 1);

        std::pair<size_t, size_t> rep_ids;
        rep_ids.first  = dist(m_rng);
        rep_ids.second = rep_ids.first;
        while (rep_ids.second == rep_ids.first) {
          rep_ids.second = dist(m_rng);
        }

        // Asynchronously broadcast the IDs of the needed reps and let them be
        // distributed from there
        m_comm.async_bcast(
            [](auto                                                 comm_ptr,
               ygm::ygm_ptr<const point_store<id_type, point_type>> p_data,
               const intra_level_tree_node_id_type                  i,
               const std::pair<size_t, size_t>                     &rep_ids) {
              if (rep_ids.first >= offsets_vec[i] &&
                  rep_ids.first < offsets_vec[i] + num_local_points_vec[i]) {
                comm_ptr->async_bcast(
                    [](const intra_level_tree_node_id_type i,
                       const point_type                   &rep) {
                      s_candidate_tree_level[i].left_rep = rep;
                    },
                    i,
                    p_data->at(
                        s_local_tree_level_id_splits[i][rep_ids.first -
                                                        offsets_vec[i]]));
              }
              if (rep_ids.second >= offsets_vec[i] &&
                  rep_ids.second < offsets_vec[i] + num_local_points_vec[i]) {
                comm_ptr->async_bcast(
                    [](const intra_level_tree_node_id_type i,
                       const point_type                   &rep) {
                      s_candidate_tree_level[i].right_rep = rep;
                    },
                    i,
                    p_data->at(
                        s_local_tree_level_id_splits[i][rep_ids.second -
                                                        offsets_vec[i]]));
              }
            },
            p_data, i, rep_ids);
      }
    }

    m_comm.barrier();

    // Calculate distance between reps for use in theta calculations
    for (intra_level_tree_node_id_type i = 0; i < candidate_tree_level.size();
         ++i) {
      candidate_tree_level[i].rep_dist = m_space.get_dist_func()(
          &(candidate_tree_level[i].left_rep),
          &(candidate_tree_level[i].right_rep), m_space.get_dist_func_param());
    }
  }

  /**
   * @brief Compute the theta values for all points based on proposed split
   *points for the current level
   *
   * @param data Container containing all IDs and points
   * @param point_assignments Global tree IDs of all locally-held points
   * @param candidate_tree_level Current tree level under consideration,
   *including the chosen left and right child reps
   * @param node_offset_curr_level The first tree ID for points within the
   *current level. Used for translating between global and intra_level tree IDs
   * @param num_nodes_curr_level The number of tree nodes in the current level
   *being constructed
   *
   * @return A vector containing vectors of all locally-computed theta values,
   *indexed by intra_level_tree_node_id_type values
   **/
  std::vector<std::vector<theta_type>> compute_split_thetas(
      const point_store<id_type, point_type> &data,
      const std::unordered_map<id_type, global_tree_node_id_type>
                                        &point_assignments,
      const std::vector<tree_node_type> &candidate_tree_level,
      global_tree_node_id_type           node_offset_curr_level,
      global_tree_node_id_type           num_nodes_curr_level) {
    std::vector<std::vector<theta_type>> curr_level_thetas(
        num_nodes_curr_level);

    for (auto point_iter = data.begin(); point_iter != data.end();
         ++point_iter) {
      const auto &[id, point]                       = *point_iter;
      global_tree_node_id_type      curr_assignment = point_assignments.at(id);
      intra_level_tree_node_id_type candidate_node_id =
          static_cast<intra_level_tree_node_id_type>(curr_assignment -
                                                     node_offset_curr_level);

      if (candidate_node_id < candidate_tree_level.size()) {
        const tree_node_type &tree_node =
            candidate_tree_level[candidate_node_id];

        theta_type point_theta = compute_theta(point, tree_node);

        curr_level_thetas[candidate_node_id].push_back(point_theta);
      }
    }

    return curr_level_thetas;
  }

  ygm::comm &m_comm;

  hnswlib::SpaceInterface<distance_type> &m_space;

  std::vector<tree_node_type> m_tree;

  uint32_t                 m_num_partitions;
  global_tree_node_id_type m_num_tree_nodes;
  tree_level_id_type       m_num_levels;
  uint16_t                 m_candidate_level_trials = 10;

  std::mt19937_64 m_rng;
};
}  // namespace saltatlas
