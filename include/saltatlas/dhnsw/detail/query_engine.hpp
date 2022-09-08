// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/detail/ygm_ptr.hpp>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename Point, typename Partitioner>
class query_engine_impl {
 public:
  class query_controller {
   public:
    query_controller() = delete;

    query_controller(
        const Point &q, const int k, const int max_hops, const int voronoi_rank,
        const int                     initial_num_queries,
        const std::vector<std::byte> &packed_callback,
        ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>> e)
        : m_query_point(q),
          m_k(k),
          m_max_hops(max_hops),
          m_initial_num_queries(initial_num_queries),
          m_voronoi_rank(voronoi_rank),
          m_queries_spawned{0},
          m_queries_returned{0},
          m_current_hops{0},
          m_num_callbacks{0},
          engine{e} {
      add_callback(packed_callback);
    };

    void start_query() {
      /*
std::vector<size_t> closest_seeds;
engine->m_dist_index_impl_ptr->find_approx_closest_seeds(
m_query_point, m_initial_num_queries, closest_seeds);
                      */

      std::vector<size_t> point_partitions =
          engine->m_dist_index_impl_ptr->partitioner().find_point_partitions(
              m_query_point, m_initial_num_queries);

      for (const auto &cell : point_partitions) {
        queue_next_cell(cell);
      }
      start_query_round();
    }

    bool has_returned() const { return m_complete; }

    DistType const neighbors_max_distance() {
      DistType to_return;
      if (m_nearest_neighbors.size() < m_k) {
        to_return = std::numeric_limits<DistType>::max();
      } else {
        to_return = (--m_nearest_neighbors.end())->first;
      }
      return to_return;
    }

    void add_callback(const std::vector<std::byte> &packed_lambda) {
      m_callbacks.insert(m_callbacks.end(), packed_lambda.begin(),
                         packed_lambda.end());
      m_num_callbacks++;
    }

   private:
    void update_nearest_neighbors(
        const std::multimap<DistType, size_t> &returned_neighbors) {
      if (m_nearest_neighbors.size() > 0) {
        merge_nearest_neighbors(returned_neighbors);
      } else {
        m_nearest_neighbors = std::move(returned_neighbors);
      }
    }

    void merge_nearest_neighbors(
        const std::multimap<DistType, size_t> &returned_neighbors) {
      auto returned_neighbor_iter = returned_neighbors.begin();

      // Add neighbors while new points are closer than my furthest current
      // neighbor, or my neighbor list isn't full
      while (((returned_neighbor_iter->first <
               (--m_nearest_neighbors.end())->first) ||
              m_nearest_neighbors.size() < m_k) &&
             returned_neighbor_iter != returned_neighbors.end()) {
        m_nearest_neighbors.insert(*returned_neighbor_iter);
        if (m_nearest_neighbors.size() > m_k) {
          m_nearest_neighbors.erase(--m_nearest_neighbors.end());
        }
        ++returned_neighbor_iter;
      }
    }

    void queue_next_cells(const std::set<size_t> &ngbr_cells) {
      for (auto &cell : ngbr_cells) {
        queue_next_cell(cell);
      }
    }

    void queue_next_cell(const size_t cell) {
      if (m_queried_cells.find(cell) == m_queried_cells.end()) {
        m_next_cells.insert(cell);
      }
    }

    void start_query_round() {
      // Swapping for new set and spawning queries in separate loop from when
      // they are inserted into m_queried_cells to avoid issues with an incoming
      // async updating these structures
      std::set<size_t> cells_set;
      cells_set.swap(m_next_cells);
      m_queries_spawned += cells_set.size();
      for (auto &cell : cells_set) {
        auto insert_ret = m_queried_cells.insert(cell);
        if (insert_ret.second == false) {
          std::cout << "Querying same cell..." << std::endl;
        }
      }
      for (auto &cell : cells_set) {
        spawn_cell_query(cell, m_k, m_voronoi_rank);
      }
      // m_next_cells.clear();
    }

    void complete_query_round() {
      if (++m_current_hops > m_max_hops || m_next_cells.empty()) {
        m_complete = true;
        complete_query();
      } else {
        start_query_round();
      }
    }

    void complete_query() {
      cereal::YGMInputArchive iarchive(m_callbacks.data(), m_callbacks.size());
      for (int i = 0; i < m_num_callbacks; ++i) {
        engine->deserialize_lambda(iarchive, m_query_point, m_nearest_neighbors,
                                   engine);
      }
      engine->m_query_controllers.erase(m_query_point);
      return;
    }

    void spawn_cell_query(const size_t cell, const int k,
                          const int voronoi_rank) {
      auto cell_query_lambda = [](auto mailbox, int from, auto engine,
                                  const Point &q, const size_t s_cell,
                                  const DistType max_dist, const int s_k,
                                  const int s_voronoi_rank) {
        int local_cell = engine->local_cell_index(s_cell);

        std::priority_queue<std::pair<DistType, hnswlib::labeltype>>
            nearest_neighbors_pq =
                engine->m_dist_index_impl_ptr->get_cell_hnsw(s_cell).searchKnn(
                    &q, s_k);

        std::set<size_t>                ngbr_cells;
        std::multimap<DistType, size_t> nearest_neighbors;

        // Loop over neighbors until priority queue is empty
        // Cannot stop when dist >= max_dist because priority queue is in
        // decreasing order
        while ((nearest_neighbors_pq.size() > 0)) {
          auto neighbor = nearest_neighbors_pq.top().second;
          auto dist     = nearest_neighbors_pq.top().first;
          if (dist < max_dist) {
            nearest_neighbors.insert({dist, neighbor});
            const auto &pointed_to_cells =
                engine->m_dist_index_impl_ptr->get_cell_pointers(neighbor);
            for (int i = 0; i < s_voronoi_rank; ++i) {
              ngbr_cells.insert(pointed_to_cells[i]);
            }
          }
          nearest_neighbors_pq.pop();
        }

        // Query found potential closest neighbors
        auto query_response_lambda =
            [](auto mailbox,
               ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>
                            engine,
               const Point &q, std::multimap<DistType, size_t> nearest_ngbrs,
               std::set<size_t> new_cells) {
              // Look up controller for returning query
              auto &query_controller =
                  engine->m_query_controllers.find(q)->second;

              query_controller.update_nearest_neighbors(nearest_ngbrs);
              query_controller.queue_next_cells(new_cells);

              if (++query_controller.m_queries_returned ==
                  query_controller.m_queries_spawned) {
                query_controller.complete_query_round();
              }

              return;
            };

        // Query did not return any potential closest neighbors
        auto empty_query_response_lambda =
            [](auto mailbox,
               ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>
                            engine,
               const Point &q) {
              // Look up controller for returning query
              auto &query_controller =
                  engine->m_query_controllers.find(q)->second;

              if (++query_controller.m_queries_returned ==
                  query_controller.m_queries_spawned) {
                query_controller.complete_query_round();
              }

              return;
            };

        if (ngbr_cells.size() == 0) {
          mailbox->async(from, empty_query_response_lambda, engine->pthis, q);
        } else {
          mailbox->async(from, query_response_lambda, engine->pthis, q,
                         nearest_neighbors, ngbr_cells);
        }

        return;
      };

      DistType max_distance = neighbors_max_distance();

      int dest = engine->m_dist_index_impl_ptr->cell_owner(cell);
      engine->m_comm->async(dest, cell_query_lambda, engine->m_comm->rank(),
                            engine->pthis, m_query_point, cell, max_distance, k,
                            voronoi_rank);
    }

    bool m_complete = false;  // Could compare m_queries_spawned vs
                              // m_queries_returned instead, but might check
                              // between round finishing and next round starting
    Point                           m_query_point;
    int                             m_k;
    int                             m_max_hops;
    int                             m_initial_num_queries;
    int                             m_voronoi_rank;
    int                             m_queries_spawned;
    int                             m_queries_returned;
    int                             m_current_hops;
    std::set<size_t>                m_queried_cells;
    std::set<size_t>                m_next_cells;
    std::multimap<DistType, size_t> m_nearest_neighbors;

    std::vector<std::byte> m_callbacks;
    int                    m_num_callbacks;

    ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>> engine;
  };

  query_engine_impl(dhnsw_impl<DistType, Point, Partitioner> *g)
      : m_comm(&g->comm()),
        m_dist_index_impl_ptr(g),
        pthis(g->comm().make_ygm_ptr(*this)){};

  ygm::comm &comm() { return *m_comm; }

  template <typename Callback, typename... CallbackArgs>
  void query(const Point &query_pt, const int k, const int max_hops,
             const int voronoi_rank, const int initial_num_queries, Callback c,
             const CallbackArgs &...args) {
    const auto packed_lambda = serialize_lambda(c, args...);
    m_comm->async(
        controller_owner(query_pt),
        [](auto comm, const Point &q, const int s_k, const int s_max_hops,
           const int s_voronoi_rank, const int s_initial_num_queries,
           const std::vector<std::byte> &packed_lambda, auto pthis) {
          pthis->initiate_query(q, s_k, s_max_hops, s_voronoi_rank,
                                s_initial_num_queries, packed_lambda);
        },
        query_pt, k, max_hops, voronoi_rank, initial_num_queries, packed_lambda,
        pthis);
  }

  int local_cell_index(const int cell) const {
    return m_dist_index_impl_ptr->local_cell_index(cell);
  }

  int controller_owner(const Point &q) const {
    /*
std::vector<size_t> closest_seeds;
m_dist_index_impl_ptr->find_approx_closest_seeds(q, 1, closest_seeds);
return m_dist_index_impl_ptr->cell_owner(closest_seeds[0]);
    */
    std::vector<size_t> partitions =
        m_dist_index_impl_ptr->partitioner().find_point_partitions(q, 1);
    return m_dist_index_impl_ptr->cell_owner(partitions[0]);
  }

 private:
  void initiate_query(const Point &q, int k, int max_hops, int voronoi_rank,
                      int                           initial_num_queries,
                      const std::vector<std::byte> &packed_lambda) {
    // Check arguments
    if (k < 1) {
      std::cerr << "Cannot specify a non-positive number of neighbors to query"
                << std::endl;
      exit(1);
    }
    if (max_hops < 0) {
      std::cerr << "Cannot specify a negative number of hops to take"
                << std::endl;
      exit(1);
    }
    if (voronoi_rank < 0) {
      std::cerr << "Cannot specify a negative number for the Voronoi rank"
                << std::endl;
      exit(1);
    }
    if (initial_num_queries < 1) {
      std::cerr
          << "Cannot specify a nonpositive number of initial queries to perform"
          << std::endl;
      exit(1);
    }

    // Create controller record locally
    auto it = m_query_controllers.find(q);
    if (it == m_query_controllers.end()) {
      auto insert_ret = m_query_controllers.insert(
          {q, query_controller(q, k, max_hops, voronoi_rank,
                               initial_num_queries, packed_lambda, pthis)});
      insert_ret.first->second.start_query();
    } else {
      (*it).second.add_callback(packed_lambda);
    }
  }

  // Stolen from YGM for lambda serialization purposes
  static void reference() {}

  template <typename Lambda, typename... PackArgs>
  std::vector<std::byte> serialize_lambda(Lambda l, const PackArgs &...args) {
    std::vector<std::byte>        to_return;
    const std::tuple<PackArgs...> tuple_args(
        std::forward<const PackArgs>(args)...);
    assert(sizeof(Lambda) == 1);

    void (*fun_ptr)(
        const Point &, const std::multimap<DistType, size_t> &,
        ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>,
        cereal::YGMInputArchive &) =
        [](const Point                           &query_pt,
           const std::multimap<DistType, size_t> &nearest_neighbors,
           ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>
                                    query_engine_ptr,
           cereal::YGMInputArchive &bia) {
          std::tuple<PackArgs...> ta;
          bia(ta);
          Lambda *pl;
          auto    t1 =
              std::make_tuple(query_pt, nearest_neighbors, query_engine_ptr);
          std::apply(*pl, std::tuple_cat(t1, ta));
        };

    cereal::YGMOutputArchive oarchive(to_return);  // Create an output archive
                                                   // // oarchive(fun_ptr);
    int64_t iptr = (int64_t)fun_ptr - (int64_t)&reference;
    oarchive(iptr, tuple_args);

    return to_return;
  }

  void deserialize_lambda(
      cereal::YGMInputArchive &iarchive, const Point &query_pt,
      const std::multimap<DistType, size_t> &nearest_neighbors,
      ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>
          query_engine_ptr) {
    int64_t iptr;
    iarchive(iptr);
    iptr += (int64_t)&reference;
    void (*fun_ptr)(
        const Point &, const std::multimap<DistType, size_t> &,
        ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>>,
        cereal::YGMInputArchive &);
    memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
    fun_ptr(query_pt, nearest_neighbors, query_engine_ptr, iarchive);
  }

  std::map<Point, query_controller> m_query_controllers;

  ygm::comm                                             *m_comm;
  ygm::ygm_ptr<dhnsw_impl<DistType, Point, Partitioner>> m_dist_index_impl_ptr;
  ygm::ygm_ptr<query_engine_impl<DistType, Point, Partitioner>> pthis;
};

}  // namespace dhnsw_detail
}  // namespace saltatlas
