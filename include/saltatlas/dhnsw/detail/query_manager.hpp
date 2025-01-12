// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/comm.hpp>

#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/neighbor_cereal.hpp>

#include <limits>
#include <queue>

namespace saltatlas {
struct dhnsw_query_params {
  int k;
  int max_hops        = 10;
  int initial_queries = 1;
};

namespace dhnsw_detail {

template <typename PointStore, typename Partitioner, typename LocalSearch,
          typename PartitionRecommender, typename PartitionMapper>
class query_manager {
 public:
  using id_type           = PointStore::id_type;
  using distance_type     = LocalSearch::distance_type;
  using point_type        = PointStore::point_type;
  using partition_id_type = uint32_t;

  using point_store_type           = PointStore;
  using partitioner_type           = Partitioner;
  using local_search_type          = LocalSearch;
  using partition_recommender_type = PartitionRecommender;
  using partition_mapper_type      = PartitionMapper;
  using self_type = query_manager<PointStore, Partitioner, LocalSearch,
                                  PartitionRecommender, PartitionMapper>;

 private:
  struct query_tracker;
  struct query_functor;
  struct response_functor;

 public:
  query_manager(ygm::comm &c, const point_store_type &point_store,
                const partitioner_type           &partitioner,
                const local_search_type          &local_search,
                const partition_recommender_type &partition_recommender,
                const partition_mapper_type &mapper, dhnsw_query_params p)
      : m_comm(c),
        m_point_store(point_store),
        m_partitioner(partitioner),
        m_local_search(local_search),
        m_partition_recommender(partition_recommender),
        m_mapper(mapper),
        m_query_params(p),
        pthis(this) {
    pthis.check(m_comm);
  }

  template <typename QueryIterator>
  std::vector<std::vector<detail::neighbor<id_type, distance_type>>> query(
      QueryIterator queries_begin, QueryIterator queries_end) {
    std::vector<std::vector<detail::neighbor<id_type, distance_type>>>
        to_return(std::distance(queries_begin, queries_end));

    size_t query_index{0};
    for (auto query_iter = queries_begin; query_iter != queries_end;
         ++query_iter) {
      const auto &point = *(query_iter);

      // TODO: Need to use leaf_hnsw to fill out remaining initial queries
      const auto partition_id = m_partitioner.find_partition(point);

      m_trackers.resize(std::distance(queries_begin, queries_end));

      auto &tracker   = m_trackers[query_index];
      tracker.p_point = &point;
      tracker.partitions_to_query.push_back(partition_id);
      start_next_query_round(query_index);

      ++query_index;
    }

    m_comm.barrier();

    for (int i = 0; i < m_trackers.size(); ++i) {
      query_tracker &tracker = m_trackers[i];

      assert(tracker.current_nn.size() <= m_query_params.k);
      to_return[i].resize(tracker.current_nn.size());
      size_t pos = to_return[i].size() - 1;
      while (not tracker.current_nn.empty()) {
        assert(pos >= 0);
        const detail::neighbor<id_type, distance_type> &nn =
            tracker.current_nn.top();
        to_return[i][pos] = nn;
        tracker.current_nn.pop();
        --pos;
      }
    }

    return to_return;
  }

 private:
  void start_next_query_round(const int local_query_id) {
    query_tracker &tracker = m_trackers[local_query_id];

    ++tracker.current_hops;

    if (tracker.current_hops > m_query_params.max_hops) {
      tracker.p_point = nullptr;
      tracker.queried_partitions.clear();
      tracker.partitions_to_query.clear();

      return;
    }

    // Remove duplicate queries before beginning
    std::sort(tracker.partitions_to_query.begin(),
              tracker.partitions_to_query.end());
    tracker.partitions_to_query.erase(
        std::unique(tracker.partitions_to_query.begin(),
                    tracker.partitions_to_query.end()),
        tracker.partitions_to_query.end());

    // Add partitions that will be queried to already queried partitions before
    // starting queries to avoid race conditions...
    const size_t prev_size = tracker.queried_partitions.size();
    tracker.queried_partitions.resize(tracker.queried_partitions.size() +
                                      tracker.partitions_to_query.size());
    std::memcpy(&tracker.queried_partitions[prev_size],
                tracker.partitions_to_query.data(),
                tracker.partitions_to_query.size() * sizeof(partition_id_type));

    // Swap to a temporary variable to avoid race conditions
    std::vector<partition_id_type> tmp_to_query;
    tmp_to_query.swap(tracker.partitions_to_query);

    distance_type max_dist;
    if (tracker.current_nn.size() > 0) {
      max_dist = tracker.current_nn.top().distance;
    } else {
      max_dist = std::numeric_limits<distance_type>::max();
    }
    tracker.outstanding_queries = tmp_to_query.size();
    for (const auto &partition : tmp_to_query) {
      int dest_rank = m_mapper.logical_to_physical(partition);
      m_comm.async(dest_rank, query_functor(), pthis, *tracker.p_point,
                   partition, m_comm.rank(), local_query_id, max_dist);
    }
  }

  struct query_tracker {
    const point_type              *p_point;
    std::vector<partition_id_type> queried_partitions;
    std::vector<partition_id_type> partitions_to_query;
    int                            current_hops;
    int                            outstanding_queries;
    std::priority_queue<detail::neighbor<id_type, distance_type>> current_nn;
  };

  struct query_functor {
    void operator()(ygm::ygm_ptr<self_type> pthis, const point_type &point,
                    const partition_id_type partition, const int sender,
                    const int           local_query_id,
                    const distance_type max_distance) {
      const auto local_partition_id =
          pthis->m_mapper.logical_to_local(partition);

      auto local_nn = pthis->m_local_search.search(
          point, pthis->m_query_params.k, local_partition_id);

      std::vector<detail::neighbor<id_type, distance_type>> neighbor_vec;
      std::vector<partition_id_type>                        neighbor_partitions;
      while (not local_nn.empty()) {
        const auto &[dist, ngbr_id] = local_nn.top();

        // hnswlib returns std::priority_queue with default comparison operator,
        // which keeps largest distances at top have to iterate over entire
        // priority queue to find best neighbors
        if (dist <= max_distance) {
          neighbor_vec.emplace_back(ngbr_id, dist);

          // Look-up new neighbor's nearby partitions
          const auto &recommended_partitions =
              pthis->m_partition_recommender.at(ngbr_id);
          const size_t prev_size = neighbor_partitions.size();
          neighbor_partitions.resize(neighbor_partitions.size() +
                                     recommended_partitions.size());
          std::memcpy(
              &neighbor_partitions[prev_size], recommended_partitions.data(),
              recommended_partitions.size() * sizeof(partition_id_type));
        }

        local_nn.pop();
      }

      // Remove duplicate partition recommendations
      std::sort(neighbor_partitions.begin(), neighbor_partitions.end());
      neighbor_partitions.erase(
          std::unique(neighbor_partitions.begin(), neighbor_partitions.end()),
          neighbor_partitions.end());

      pthis->m_comm.async(sender, response_functor(), pthis, local_query_id,
                          neighbor_vec, neighbor_partitions);
    }
  };

  struct response_functor {
    void operator()(ygm::ygm_ptr<self_type> pthis, const int local_query_id,
                    const std::vector<detail::neighbor<id_type, distance_type>>
                                                         neighbor_dist_id_vec,
                    const std::vector<partition_id_type> neighbor_partitions) {
      auto &tracker = pthis->m_trackers[local_query_id];

      for (const auto &dist_id : neighbor_dist_id_vec) {
        tracker.current_nn.push(dist_id);
      }
      while (tracker.current_nn.size() > pthis->m_query_params.k) {
        tracker.current_nn.pop();
      }

      for (const partition_id_type &ngbr_partition : neighbor_partitions) {
        if (not std::binary_search(tracker.queried_partitions.begin(),
                                   tracker.queried_partitions.end(),
                                   ngbr_partition)) {
          tracker.partitions_to_query.push_back(ngbr_partition);
        }
      }

      --tracker.outstanding_queries;

      if (tracker.outstanding_queries == 0) {
        pthis->start_next_query_round(local_query_id);
      }
    }
  };

  ygm::comm &m_comm;

  const point_store_type           &m_point_store;
  const partitioner_type           &m_partitioner;
  const local_search_type          &m_local_search;
  const partition_recommender_type &m_partition_recommender;
  const partition_mapper_type
      &m_mapper;  // TODO: Can remove m_mapper here if all partitions throughout
                  // code are treated as partition_locators that have rank and
                  // local_partition built in

  std::vector<query_tracker> m_trackers;

  const dhnsw_query_params m_query_params;

  ygm::ygm_ptr<self_type> pthis;
};
}  // namespace dhnsw_detail
}  // namespace saltatlas
