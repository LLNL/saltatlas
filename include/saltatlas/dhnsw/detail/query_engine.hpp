// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/detail/ygm_ptr.hpp>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
class query_engine_impl {
 public:
  using dist_t        = DistType;
  using index_t       = IndexType;
  using point_t       = Point;
  using partitioner_t = Partitioner<dist_t, index_t, point_t>;
  using query_engine_impl_t =
      query_engine_impl<dist_t, index_t, point_t, Partitioner>;
  using dhnsw_impl_t     = dhnsw_impl<dist_t, index_t, point_t, Partitioner>;
  using dist_ngbr_mmap_t = std::multimap<dist_t, index_t>;
  using dist_ngbr_owner_map_t    = std::map<index_t, int>;
  using dist_ngbr_features_map_t = std::map<index_t, point_t>;
  using dist_ngbr_features_mmap_t =
      std::multimap<dist_t, std::pair<index_t, point_t>>;

  class query_controller {
   public:
    class with_features_tag_t {};
    static with_features_tag_t with_features_tag;

    query_controller() = delete;

    query_controller(const point_t &q, const int k, const int max_hops,
                     const int voronoi_rank, const int initial_num_queries,
                     const std::vector<std::byte>     &packed_callback,
                     ygm::ygm_ptr<query_engine_impl_t> e)
        : m_query_point(q),
          m_k(k),
          m_max_hops(max_hops),
          m_initial_num_queries(initial_num_queries),
          m_voronoi_rank(voronoi_rank),
          m_queries_spawned{0},
          m_queries_returned{0},
          m_current_hops{0},
          m_num_callbacks{0},
          engine{e},
          m_query_with_features{false} {
      add_callback(packed_callback);
    };

    query_controller(const point_t &q, const int k, const int max_hops,
                     const int voronoi_rank, const int initial_num_queries,
                     const std::vector<std::byte>     &packed_callback,
                     ygm::ygm_ptr<query_engine_impl_t> e,
                     with_features_tag_t               features_tag)
        : m_query_point(q),
          m_k(k),
          m_max_hops(max_hops),
          m_initial_num_queries(initial_num_queries),
          m_voronoi_rank(voronoi_rank),
          m_queries_spawned{0},
          m_queries_returned{0},
          m_current_hops{0},
          m_num_callbacks{0},
          engine{e},
          m_query_with_features{true} {
      add_callback(packed_callback);
    };

    void start_query() {
      /*
std::vector<index_t> closest_seeds;
engine->m_dist_index_impl_ptr->find_approx_closest_seeds(
m_query_point, m_initial_num_queries, closest_seeds);
                      */

      std::vector<index_t> point_partitions =
          engine->m_dist_index_impl_ptr->partitioner().find_point_partitions(
              m_query_point, m_initial_num_queries);

      for (const auto &cell : point_partitions) {
        queue_next_cell(cell);
      }
      start_query_round();
    }

    bool has_returned() const { return m_complete; }

    dist_t const neighbors_max_distance() {
      dist_t to_return;
      if (m_nearest_neighbors.size() < m_k) {
        to_return = std::numeric_limits<dist_t>::max();
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
    void update_nearest_neighbors(const dist_ngbr_mmap_t &returned_neighbors,
                                  const int               owner_rank) {
      if (m_nearest_neighbors.size() > 0) {
        merge_nearest_neighbors(returned_neighbors, owner_rank);
      } else {
        m_nearest_neighbors = std::move(returned_neighbors);

        for (const auto &[dist, index] : m_nearest_neighbors) {
          m_nearest_neighbor_owners[index] = owner_rank;
        }
      }
    }

    void merge_nearest_neighbors(const dist_ngbr_mmap_t &returned_neighbors,
                                 const int               owner_rank) {
      auto returned_neighbor_iter = returned_neighbors.begin();

      // Add neighbors while new points are closer than my furthest current
      // neighbor, or my neighbor list isn't full
      while (((returned_neighbor_iter->first <
               (--m_nearest_neighbors.end())->first) ||
              m_nearest_neighbors.size() < m_k) &&
             returned_neighbor_iter != returned_neighbors.end()) {
        m_nearest_neighbor_owners[returned_neighbor_iter->second] = owner_rank;
        m_nearest_neighbors.insert(*returned_neighbor_iter);
        if (m_nearest_neighbors.size() > m_k) {
          // No longer need to know owner of neighbor being removed
          m_nearest_neighbor_owners.erase(
              (--m_nearest_neighbors.end())->second);
          m_nearest_neighbors.erase(--m_nearest_neighbors.end());
        }
        ++returned_neighbor_iter;
      }
    }

    void queue_next_cells(const std::set<index_t> &ngbr_cells) {
      for (auto &cell : ngbr_cells) {
        queue_next_cell(cell);
      }
    }

    void queue_next_cell(const index_t cell) {
      if (m_queried_cells.find(cell) == m_queried_cells.end()) {
        m_next_cells.insert(cell);
      }
    }

    void start_query_round() {
      // Swapping for new set and spawning queries in separate loop from when
      // they are inserted into m_queried_cells to avoid issues with an incoming
      // async updating these structures
      std::set<index_t> cells_set;
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
      if (!m_query_with_features) {
        cereal::YGMInputArchive iarchive(m_callbacks.data(),
                                         m_callbacks.size());
        for (int i = 0; i < m_num_callbacks; ++i) {
          engine->deserialize_lambda(iarchive, m_query_point,
                                     m_nearest_neighbors, engine);
        }
        engine->m_query_controllers.erase(m_query_point);
        return;
      } else {
        auto get_neighbor_features_lambda = [](const int controller_owner,
                                               auto engine, const point_t &q,
                                               const index_t ngbr_index) {
          auto neighbor_features_response_lambda = [](auto           engine,
                                                      const point_t &q,
                                                      const index_t  ngbr_index,
                                                      const point_t &ngbr) {
            auto &query_controller =
                engine->m_query_controllers.find(q)->second;

            query_controller.m_nearest_neighbor_features[ngbr_index] = ngbr;

            if (query_controller.m_nearest_neighbor_features.size() ==
                query_controller.m_nearest_neighbors.size()) {
              dist_ngbr_features_mmap_t nn_mmap;
              for (const auto &dist_index :
                   query_controller.m_nearest_neighbors) {
                const auto &[ngbr_dist, ngbr_index] = dist_index;
                nn_mmap.insert(std::make_pair(
                    ngbr_dist,
                    std::make_pair(
                        ngbr_index,
                        query_controller
                            .m_nearest_neighbor_features[ngbr_index])));
              }

              cereal::YGMInputArchive iarchive(
                  query_controller.m_callbacks.data(),
                  query_controller.m_callbacks.size());
              for (int i = 0; i < query_controller.m_num_callbacks; ++i) {
                // TODO: This is a copy of deserialize_lambda with a different
                // multimap type to accomodate feature vectors...
                int64_t iptr;
                iarchive(iptr);
                iptr += (int64_t)&reference;
                void (*fun_ptr)(const point_t &,
                                const dist_ngbr_features_mmap_t &,
                                ygm::ygm_ptr<query_engine_impl_t>,
                                cereal::YGMInputArchive &);
                memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
                fun_ptr(q, nn_mmap, engine, iarchive);
              }
              engine->m_query_controllers.erase(query_controller.m_query_point);
            }
          };

          const auto &ngbr_pt =
              engine->m_dist_index_impl_ptr->get_point(ngbr_index);

          engine->m_comm->async(controller_owner,
                                neighbor_features_response_lambda,
                                engine->pthis, q, ngbr_index, ngbr_pt);
        };

        for (const auto &[idx, owner_rank] : m_nearest_neighbor_owners) {
          engine->m_comm->async(owner_rank, get_neighbor_features_lambda,
                                engine->m_comm->rank(), engine->pthis,
                                m_query_point, idx);
        }
      }
    }

    void spawn_cell_query(const index_t cell, const int k,
                          const int voronoi_rank) {
      auto cell_query_lambda = [](auto mailbox, int from, auto engine,
                                  const point_t &q, const index_t s_cell,
                                  const dist_t max_dist, const int s_k,
                                  const int s_voronoi_rank) {
        int local_cell = engine->local_cell_index(s_cell);

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
            nearest_neighbors_pq =
                engine->m_dist_index_impl_ptr->get_cell_hnsw(s_cell).searchKnn(
                    &q, s_k);

        std::set<index_t> ngbr_cells;
        dist_ngbr_mmap_t  nearest_neighbors;

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
            [](auto mailbox, int from, ygm::ygm_ptr<query_engine_impl_t> engine,
               const point_t &q, dist_ngbr_mmap_t nearest_ngbrs,
               std::set<index_t> new_cells) {
              // Look up controller for returning query
              auto &query_controller =
                  engine->m_query_controllers.find(q)->second;

              query_controller.update_nearest_neighbors(nearest_ngbrs, from);
              query_controller.queue_next_cells(new_cells);

              if (++query_controller.m_queries_returned ==
                  query_controller.m_queries_spawned) {
                query_controller.complete_query_round();
              }

              return;
            };

        // Query did not return any potential closest neighbors
        auto empty_query_response_lambda =
            [](auto mailbox, ygm::ygm_ptr<query_engine_impl_t> engine,
               const point_t &q) {
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
          mailbox->async(from, query_response_lambda, engine->m_comm->rank(),
                         engine->pthis, q, nearest_neighbors, ngbr_cells);
        }

        return;
      };

      dist_t max_distance = neighbors_max_distance();

      int dest = engine->m_dist_index_impl_ptr->cell_owner(cell);
      engine->m_comm->async(dest, cell_query_lambda, engine->m_comm->rank(),
                            engine->pthis, m_query_point, cell, max_distance, k,
                            voronoi_rank);
    }

    bool m_complete = false;  // Could compare m_queries_spawned vs
                              // m_queries_returned instead, but might check
                              // between round finishing and next round starting
    point_t           m_query_point;
    int               m_k;
    int               m_max_hops;
    int               m_initial_num_queries;
    int               m_voronoi_rank;
    int               m_queries_spawned;
    int               m_queries_returned;
    int               m_current_hops;
    std::set<index_t> m_queried_cells;
    std::set<index_t> m_next_cells;
    dist_ngbr_mmap_t  m_nearest_neighbors;

    dist_ngbr_owner_map_t m_nearest_neighbor_owners;

    dist_ngbr_features_map_t m_nearest_neighbor_features;

    std::vector<std::byte> m_callbacks;
    int                    m_num_callbacks;

    ygm::ygm_ptr<query_engine_impl_t> engine;

    bool m_query_with_features{false};
  };

  query_engine_impl(dhnsw_impl_t *g)
      : m_comm(&g->comm()),
        m_dist_index_impl_ptr(g),
        pthis(g->comm().make_ygm_ptr(*this)){};

  ygm::comm &comm() { return *m_comm; }

  template <typename Callback, typename... CallbackArgs>
  void query(const point_t &query_pt, const int k, const int max_hops,
             const int voronoi_rank, const int initial_num_queries, Callback c,
             const CallbackArgs &...args) {
    const auto packed_lambda = serialize_lambda(c, args...);
    m_comm->async(
        controller_owner(query_pt),
        [](auto comm, const point_t &q, const int s_k, const int s_max_hops,
           const int s_voronoi_rank, const int s_initial_num_queries,
           const std::vector<std::byte> &packed_lambda, auto pthis) {
          pthis->initiate_query(q, s_k, s_max_hops, s_voronoi_rank,
                                s_initial_num_queries, packed_lambda);
        },
        query_pt, k, max_hops, voronoi_rank, initial_num_queries, packed_lambda,
        pthis);
  }

  template <typename Callback, typename... CallbackArgs>
  void query_with_features(const point_t &query_pt, const int k,
                           const int max_hops, const int voronoi_rank,
                           const int initial_num_queries, Callback c,
                           const CallbackArgs &...args) {
    const auto packed_lambda = serialize_lambda_with_features(c, args...);
    m_comm->async(
        controller_owner(query_pt),
        [](auto comm, const point_t &q, const int s_k, const int s_max_hops,
           const int s_voronoi_rank, const int s_initial_num_queries,
           const std::vector<std::byte> &packed_lambda, auto pthis) {
          pthis->initiate_query_with_features(
              q, s_k, s_max_hops, s_voronoi_rank, s_initial_num_queries,
              packed_lambda);
        },
        query_pt, k, max_hops, voronoi_rank, initial_num_queries, packed_lambda,
        pthis);
  }

  int local_cell_index(const int cell) const {
    return m_dist_index_impl_ptr->local_cell_index(cell);
  }

  int controller_owner(const point_t &q) const {
    /*
std::vector<index_t> closest_seeds;
m_dist_index_impl_ptr->find_approx_closest_seeds(q, 1, closest_seeds);
return m_dist_index_impl_ptr->cell_owner(closest_seeds[0]);
    */
    std::vector<index_t> partitions =
        m_dist_index_impl_ptr->partitioner().find_point_partitions(q, 1);
    return m_dist_index_impl_ptr->cell_owner(partitions[0]);
  }

 private:
  void initiate_query(const point_t &q, int k, int max_hops, int voronoi_rank,
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

  void initiate_query_with_features(
      const point_t &q, int k, int max_hops, int voronoi_rank,
      int initial_num_queries, const std::vector<std::byte> &packed_lambda) {
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
                               initial_num_queries, packed_lambda, pthis,
                               query_controller::with_features_tag)});
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

    void (*fun_ptr)(const point_t &, const dist_ngbr_mmap_t &,
                    ygm::ygm_ptr<query_engine_impl_t>,
                    cereal::YGMInputArchive &) =
        [](const point_t &query_pt, const dist_ngbr_mmap_t &nearest_neighbors,
           ygm::ygm_ptr<query_engine_impl_t> query_engine_ptr,
           cereal::YGMInputArchive          &bia) {
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

  template <typename Lambda, typename... PackArgs>
  std::vector<std::byte> serialize_lambda_with_features(
      Lambda l, const PackArgs &...args) {
    std::vector<std::byte>        to_return;
    const std::tuple<PackArgs...> tuple_args(
        std::forward<const PackArgs>(args)...);
    assert(sizeof(Lambda) == 1);

    void (*fun_ptr)(const point_t &, const dist_ngbr_features_mmap_t &,
                    ygm::ygm_ptr<query_engine_impl_t>,
                    cereal::YGMInputArchive &) =
        [](const point_t                    &query_pt,
           const dist_ngbr_features_mmap_t  &nearest_neighbors,
           ygm::ygm_ptr<query_engine_impl_t> query_engine_ptr,
           cereal::YGMInputArchive          &bia) {
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

  void deserialize_lambda(cereal::YGMInputArchive          &iarchive,
                          const point_t                    &query_pt,
                          const dist_ngbr_mmap_t           &nearest_neighbors,
                          ygm::ygm_ptr<query_engine_impl_t> query_engine_ptr) {
    int64_t iptr;
    iarchive(iptr);
    iptr += (int64_t)&reference;
    void (*fun_ptr)(const point_t &, const dist_ngbr_mmap_t &,
                    ygm::ygm_ptr<query_engine_impl_t>,
                    cereal::YGMInputArchive &);
    memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
    fun_ptr(query_pt, nearest_neighbors, query_engine_ptr, iarchive);
  }

  std::map<point_t, query_controller> m_query_controllers;

  ygm::comm                        *m_comm;
  ygm::ygm_ptr<dhnsw_impl_t>        m_dist_index_impl_ptr;
  ygm::ygm_ptr<query_engine_impl_t> pthis;
};

}  // namespace dhnsw_detail
}  // namespace saltatlas
