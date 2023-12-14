// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/detail/meta/functional.hpp>
#include <ygm/detail/ygm_ptr.hpp>

namespace saltatlas {
namespace dhnsw_detail {

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
class query_engine_impl {
 public:
  using dist_type     = DistType;
  using index_type    = IndexType;
  using point_type    = Point;
  using partitioner_type = Partitioner<dist_type, index_type, point_type>;
  using query_engine_impl_type =
      query_engine_impl<dist_type, index_type, point_type, Partitioner>;
  using dhnsw_impl_type =
      dhnsw_impl<dist_type, index_type, point_type, Partitioner>;
  using dist_ngbr_mmap_type          = std::multimap<dist_type, index_type>;
  using dist_ngbr_owner_map_type     = std::map<index_type, int>;
  using dist_ngbr_features_map_type  = std::map<index_type, point_type>;
  using dist_ngbr_features_mmap_type =
      std::multimap<dist_type, std::pair<index_type, point_type>>;

  using query_id_type = uint32_t;

  class query_controller {
   private:
    struct query_locator {
      int           rank;
      query_id_type local_id;

      template <typename Archive>
      void serialize(Archive &ar) {
        ar(rank, local_id);
      }
    };

   public:
    class with_features_tag_t {};
    static with_features_tag_t with_features_tag;

    query_controller() = delete;

    query_controller(const point_type &q, const int k, const int max_hops,
                     const int voronoi_rank, const int initial_num_queries,
                     const std::vector<std::byte> &packed_callback,
                     query_id_type id, ygm::ygm_ptr<query_engine_impl_type> e)
        : m_query_point(q),
          m_k(k),
          m_max_hops(max_hops),
          m_initial_num_queries(initial_num_queries),
          m_voronoi_rank(voronoi_rank),
          m_queries_spawned{0},
          m_queries_returned{0},
          m_current_hops{0},
          m_id{id},
          engine{e},
          m_query_with_features{false} {
      add_callback(packed_callback);
    };

    query_controller(const point_type &q, const int k, const int max_hops,
                     const int voronoi_rank, const int initial_num_queries,
                     const std::vector<std::byte> &packed_callback,
                     query_id_type id, ygm::ygm_ptr<query_engine_impl_type> e,
                     with_features_tag_t features_tag)
        : m_query_point(q),
          m_k(k),
          m_max_hops(max_hops),
          m_initial_num_queries(initial_num_queries),
          m_voronoi_rank(voronoi_rank),
          m_queries_spawned{0},
          m_queries_returned{0},
          m_current_hops{0},
          m_id{id},
          engine{e},
          m_query_with_features{true} {
      add_callback(packed_callback);
    };

    const point_type get_query_point() const { return m_query_point; }
    const int     get_k() const { return m_k; }
    const int     get_max_hops() const { return m_k; }
    const int get_initial_num_queries() const { return m_initial_num_queries; }
    const int get_voronoi_rank() const { return m_voronoi_rank; }
    const int get_num_local_queries() const { return m_queries_spawned; }
    const int get_num_hops() const { return m_current_hops; }
    const std::set<index_type> &get_queried_cells() const {
      return m_queried_cells;
    }
    ygm::ygm_ptr<query_engine_impl_type> get_query_engine_ptr() const {
      return engine;
    }

    std::vector<point_type> get_queried_representatives() const {
      std::vector<point_type> to_return;

      auto cells = get_queried_cells();

      for (const auto cell : cells) {
        auto &rep = get_query_engine_ptr()
                        ->m_dist_index_impl_ptr->partitioner()
                        .get_partition_representative(cell);
        to_return.push_back(rep);
      }

      return to_return;
    }

    const point_type find_point_partition_representative(
        const point_type &features) const {
      return get_query_engine_ptr()
          ->m_dist_index_impl_ptr->partitioner()
          .find_point_partition_representative(features);
    }

    const point_type find_query_partition_representative() const {
      return find_point_partition_representative(m_query_point);
    }

    void start_query() {
      std::vector<index_type> point_partitions =
          engine->m_dist_index_impl_ptr->partitioner().find_point_partitions(
              m_query_point, m_initial_num_queries);

      for (const auto &cell : point_partitions) {
        queue_next_cell(cell);
      }
      start_query_round();
    }

    bool has_returned() const { return m_complete; }

    dist_type const neighbors_max_distance() {
      dist_type to_return;
      if (m_nearest_neighbors.size() < m_k) {
        to_return = std::numeric_limits<dist_type>::max();
      } else {
        to_return = (--m_nearest_neighbors.end())->first;
      }
      return to_return;
    }

    void add_callback(const std::vector<std::byte> &packed_lambda) {
      m_callbacks.insert(m_callbacks.end(), packed_lambda.begin(),
                         packed_lambda.end());
    }

   private:
    void update_nearest_neighbors(const dist_ngbr_mmap_type &returned_neighbors,
                                  const int                  owner_rank) {
      ASSERT_RELEASE(owner_rank < engine->m_comm->size());
      if (m_nearest_neighbors.size() > 0) {
        merge_nearest_neighbors(returned_neighbors, owner_rank);
      } else {
        m_nearest_neighbors = std::move(returned_neighbors);

        for (const auto &[dist, index] : m_nearest_neighbors) {
          m_nearest_neighbor_owners[index] = owner_rank;
        }
      }
    }

    void merge_nearest_neighbors(const dist_ngbr_mmap_type &returned_neighbors,
                                 const int                  owner_rank) {
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

    void queue_next_cells(const std::set<index_type> &ngbr_cells) {
      for (auto &cell : ngbr_cells) {
        queue_next_cell(cell);
      }
    }

    void queue_next_cell(const index_type cell) {
      if (m_queried_cells.find(cell) == m_queried_cells.end()) {
        m_next_cells.insert(cell);
      }
    }

    void start_query_round() {
      // Swapping for new set and spawning queries in separate loop from when
      // they are inserted into m_queried_cells to avoid issues with an incoming
      // async updating these structures
      std::set<index_type> cells_set;
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
        engine->deserialize_lambda(iarchive, m_query_point, m_nearest_neighbors,
                                   *this);

        engine->m_query_id_recycler.return_id(m_id);
        return;
      } else {
        auto get_neighbor_features_lambda =
            [](auto engine, const query_locator locator,
               const index_type ngbr_index) {
          auto neighbor_features_response_lambda =
              [](auto engine, const query_id_type &id,
                 const index_type ngbr_index, const point_type &ngbr) {
                auto &query_controller = engine->m_query_controllers[id];

                query_controller.m_nearest_neighbor_features[ngbr_index] = ngbr;
                ASSERT_RELEASE(
                    query_controller.m_nearest_neighbor_features.size() <=
                    query_controller.m_k);

                if (query_controller.m_nearest_neighbor_features.size() ==
                    query_controller.m_nearest_neighbors.size()) {
                  dist_ngbr_features_mmap_type nn_mmap;
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

                  // TODO: This is a copy of deserialize_lambda with a
                  // different multimap type to accomodate feature vectors...
                  /*
int64_t iptr;
iarchive(iptr);
iptr += (int64_t)&reference;
void (*fun_ptr)(
const point_type &, const dist_ngbr_features_mmap_type &,
const query_controller &, cereal::YGMInputArchive &);
memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
fun_ptr(query_controller.m_query_point, nn_mmap, *this, iarchive);
                  */

                  engine->deserialize_lambda_with_features(
                      iarchive, query_controller.m_query_point, nn_mmap,
                      query_controller);

                  engine->m_query_id_recycler.return_id(id);
                }
              };

          const auto &ngbr_pt =
              engine->m_dist_index_impl_ptr->get_point(ngbr_index);

          engine->m_comm->async(locator.rank, neighbor_features_response_lambda,
                                engine->pthis, locator.local_id, ngbr_index,
                                ngbr_pt);
        };

        query_locator locator{engine->m_comm->rank(), m_id};
        for (const auto &[idx, owner_rank] : m_nearest_neighbor_owners) {
          ASSERT_RELEASE(owner_rank < engine->m_comm->size());
          engine->m_comm->async(owner_rank, get_neighbor_features_lambda,
                                engine->pthis, locator, idx);
        }
      }
    }

    void spawn_cell_query(const index_type cell, const int k,
                          const int voronoi_rank) {
      auto cell_query_lambda = [](auto engine, const point_type &q,
                                  const index_type s_cell,
                                  const dist_type max_dist, const int s_k,
                                  const int s_voronoi_rank,
                                  const query_locator locator) {
        int local_cell = engine->local_cell_index(s_cell);

        std::priority_queue<std::pair<dist_type, hnswlib::labeltype>>
            nearest_neighbors_pq =
                engine->m_dist_index_impl_ptr->get_cell_hnsw(s_cell).searchKnn(
                    &q, s_k);

        std::set<index_type> ngbr_cells;
        dist_ngbr_mmap_type  nearest_neighbors;

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
            [](ygm::ygm_ptr<query_engine_impl_type> engine,
               const query_id_type id, dist_ngbr_mmap_type nearest_ngbrs,
               std::set<index_type> new_cells, const int queried_rank) {
              // Look up controller for returning query
              auto &query_controller = engine->m_query_controllers[id];

              query_controller.update_nearest_neighbors(nearest_ngbrs,
                                                        queried_rank);
              query_controller.queue_next_cells(new_cells);

              if (++query_controller.m_queries_returned ==
                  query_controller.m_queries_spawned) {
                query_controller.complete_query_round();
              }

              return;
            };

        // Query did not return any potential closest neighbors
        auto empty_query_response_lambda =
            [](ygm::ygm_ptr<query_engine_impl_type> engine,
               const query_id_type                 &id) {
              // Look up controller for returning query
              auto &query_controller = engine->m_query_controllers[id];

              if (++query_controller.m_queries_returned ==
                  query_controller.m_queries_spawned) {
                query_controller.complete_query_round();
              }

              return;
            };

        if (ngbr_cells.size() == 0) {
          engine->m_comm->async(locator.rank, empty_query_response_lambda,
                                engine->pthis, locator.local_id);
        } else {
          engine->m_comm->async(locator.rank, query_response_lambda,
                                engine->pthis, locator.local_id,
                                nearest_neighbors, ngbr_cells,
                                engine->m_comm->rank());
        }

        return;
      };

      dist_type max_distance = neighbors_max_distance();

      int           dest = engine->m_dist_index_impl_ptr->cell_owner(cell);
      query_locator locator{engine->m_comm->rank(), m_id};
      engine->m_comm->async(dest, cell_query_lambda, engine->pthis,
                            m_query_point, cell, max_distance, k, voronoi_rank,
                            locator);
    }

    bool m_complete = false;  // Could compare m_queries_spawned vs
                              // m_queries_returned instead, but might check
                              // between round finishing and next round starting
    point_type           m_query_point;
    int               m_k;
    int               m_max_hops;
    int               m_initial_num_queries;
    int               m_voronoi_rank;
    int               m_queries_spawned;
    int               m_queries_returned;
    int               m_current_hops;
    std::set<index_type> m_queried_cells;
    std::set<index_type> m_next_cells;
    dist_ngbr_mmap_type  m_nearest_neighbors;

    dist_ngbr_owner_map_type m_nearest_neighbor_owners;

    dist_ngbr_features_map_type m_nearest_neighbor_features;

    std::vector<std::byte> m_callbacks;

    query_id_type m_id;

    ygm::ygm_ptr<query_engine_impl_type> engine;

    bool m_query_with_features{false};
  };

  template <typename T>
  class id_recycler {
   public:
    id_recycler() {}

    bool has_id_available() { return m_available_ids.size() > 0; }

    T get_id() {
      ASSERT_RELEASE(m_available_ids.size() > 0);

      T id = m_available_ids.back();
      m_available_ids.pop_back();

      return id;
    }

    void return_id(const T id) { m_available_ids.push_back(id); }

   private:
    std::vector<T> m_available_ids;
  };

  query_engine_impl(dhnsw_impl_type *g)
      : m_comm(&g->comm()),
        m_dist_index_impl_ptr(g),
        pthis(g->comm().make_ygm_ptr(*this)){};

  ~query_engine_impl() { m_comm->barrier(); }

  ygm::comm &comm() { return *m_comm; }

  template <typename Callback, typename... CallbackArgs>
  void query(const point_type &query_pt, const int k, const int max_hops,
             const int voronoi_rank, const int initial_num_queries, Callback c,
             const CallbackArgs &...args) {
    const auto packed_lambda = serialize_lambda(c, args...);

    initiate_query(query_pt, k, max_hops, voronoi_rank, initial_num_queries,
                   packed_lambda);
  }

  template <typename Callback, typename... CallbackArgs>
  void query_with_features(const point_type &query_pt, const int k,
                           const int max_hops, const int voronoi_rank,
                           const int initial_num_queries, Callback c,
                           const CallbackArgs &...args) {
    const auto packed_lambda = serialize_lambda_with_features(c, args...);

    initiate_query_with_features(query_pt, k, max_hops, voronoi_rank,
                                 initial_num_queries, packed_lambda);
  }

  int local_cell_index(const int cell) const {
    return m_dist_index_impl_ptr->local_cell_index(cell);
  }

 private:
  void initiate_query(const point_type &q, int k, int max_hops,
                      int voronoi_rank, int initial_num_queries,
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

    if (m_query_id_recycler.has_id_available()) {
      const auto id = m_query_id_recycler.get_id();

      m_query_controllers[id] =
          query_controller(q, k, max_hops, voronoi_rank, initial_num_queries,
                           packed_lambda, id, pthis);

      m_query_controllers[id].start_query();
    } else {
      const query_id_type id = m_query_controllers.size();

      m_query_controllers.emplace_back(q, k, max_hops, voronoi_rank,
                                       initial_num_queries, packed_lambda, id,
                                       pthis);

      m_query_controllers[id].start_query();
    }
  }

  void initiate_query_with_features(
      const point_type &q, int k, int max_hops, int voronoi_rank,
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

    if (m_query_id_recycler.has_id_available()) {
      const auto id = m_query_id_recycler.get_id();

      m_query_controllers[id] = query_controller(
          q, k, max_hops, voronoi_rank, initial_num_queries, packed_lambda, id,
          pthis, query_controller::with_features_tag);

      m_query_controllers[id].start_query();
    } else {
      const auto id = m_query_controllers.size();

      query_controller cont(q, k, max_hops, voronoi_rank, initial_num_queries,
                            packed_lambda, id, pthis,
                            query_controller::with_features_tag);

      m_query_controllers.push_back(std::move(cont));

      m_query_controllers[id].start_query();
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

    void (*fun_ptr)(const point_type &, const dist_ngbr_mmap_type &,
                    const query_controller &, cereal::YGMInputArchive &) =
        [](const point_type &query_pt,
           const dist_ngbr_mmap_type &nearest_neighbors,
           const query_controller &controller, cereal::YGMInputArchive &bia) {
          std::tuple<PackArgs...> ta;
          bia(ta);
          Lambda *pl;
          auto    t1 = std::make_tuple(query_pt, nearest_neighbors);
          ygm::meta::apply_optional(*pl, std::make_tuple(controller),
                                    std::tuple_cat(t1, ta));
        };

    cereal::YGMOutputArchive oarchive(to_return);  // Create an output archive
    int64_t                  iptr = (int64_t)fun_ptr - (int64_t)&reference;
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

    void (*fun_ptr)(const point_type &, const dist_ngbr_features_mmap_type &,
                    const query_controller &, cereal::YGMInputArchive &) =
        [](const point_type                   &query_pt,
           const dist_ngbr_features_mmap_type &nearest_neighbors,
           const query_controller &controller, cereal::YGMInputArchive &bia) {
          std::tuple<PackArgs...> ta;
          bia(ta);
          Lambda                              *pl;
          std::tuple<const query_controller &> optional_args =
              std::make_tuple(controller);
          auto t1 = std::make_tuple(query_pt, nearest_neighbors);
          ygm::meta::apply_optional(*pl, std::make_tuple(controller),
                                    std::tuple_cat(t1, ta));
        };

    cereal::YGMOutputArchive oarchive(to_return);  // Create an output archive
    int64_t                  iptr = (int64_t)fun_ptr - (int64_t)&reference;
    oarchive(iptr, tuple_args);

    return to_return;
  }

  void deserialize_lambda(cereal::YGMInputArchive &iarchive,
                          const point_type           &query_pt,
                          const dist_ngbr_mmap_type  &nearest_neighbors,
                          const query_controller  &controller) {
    int64_t iptr;
    iarchive(iptr);
    iptr += (int64_t)&reference;
    void (*fun_ptr)(const point_type &, const dist_ngbr_mmap_type &,
                    const query_controller &, cereal::YGMInputArchive &);
    memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
    fun_ptr(query_pt, nearest_neighbors, controller, iarchive);
  }

  void deserialize_lambda_with_features(
      cereal::YGMInputArchive &iarchive, const point_type &query_pt,
      const dist_ngbr_features_mmap_type &nearest_neighbors,
      const query_controller          &controller) {
    int64_t iptr;
    iarchive(iptr);
    iptr += (int64_t)&reference;
    void (*fun_ptr)(const point_type &, const dist_ngbr_features_mmap_type &,
                    const query_controller &, cereal::YGMInputArchive &);
    memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
    fun_ptr(query_pt, nearest_neighbors, controller, iarchive);
  }

  std::vector<query_controller> m_query_controllers;
  id_recycler<query_id_type>       m_query_id_recycler;

  ygm::comm                        *m_comm;
  ygm::ygm_ptr<dhnsw_impl_type>        m_dist_index_impl_ptr;
  ygm::ygm_ptr<query_engine_impl_type> pthis;
};

}  // namespace dhnsw_detail
}  // namespace saltatlas
