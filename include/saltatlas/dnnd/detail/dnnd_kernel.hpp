// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
#define SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS 0
#endif

#ifndef SALTATLAS_DNND_SHOW_MSG_DST_STATISTICS
#define SALTATLAS_DNND_SHOW_MSG_DST_STATISTICS 0
#endif

#ifndef SALTATLAS_DNND_PRUNE_LONG_DISTANCE_MSGS
#define SALTATLAS_DNND_PRUNE_LONG_DISTANCE_MSGS 1
#endif

#ifndef SALTATLAS_DNND_PROFILE_FEATURE_MSG
#define SALTATLAS_DNND_PROFILE_FEATURE_MSG 0
#endif
#if SALTATLAS_DNND_PROFILE_FEATURE_MSG
#warning "SALTATLAS_DNND_PROFILE_FEATURE_MSG is enabled."
#endif

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if __has_include(<boost/unordered/unordered_flat_map.hpp>) \
&& __has_include(<boost/unordered/unordered_node_map.hpp>) \
&& defined(BOOST_VERSION) && BOOST_VERSION >= 108200
#ifndef SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
#define SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER 1
#endif
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#endif

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/neighbor_cereal.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/utilities/mpi.hpp>
#include <saltatlas/dnnd/detail/utilities/ygm.hpp>
#include "saltatlas/point_store.hpp"

namespace saltatlas::dndetail {

template <typename PointStore, typename Distance>
class dnnd_kernel {
 public:
  using id_type       = typename PointStore::id_type;
  using distance_type = Distance;
  using point_type    = typename PointStore::point_type;
  // Redefine point store type so that autocompletion works when writing code.
  using point_store_type =
      point_store<id_type, point_type, typename PointStore::hasher,
                  typename PointStore::equal_to,
                  typename PointStore::allocator_type>;
  using point_partitioner = std::function<int(const id_type& id)>;
  using distance_function_type =
      saltatlas::distance::distance_function_type<point_type, distance_type>;

  struct option {
    int         k{4};
    double      r{1.0};
    double      delta{0.1};
    bool        exchange_reverse_neighbors{false};
    std::size_t mini_batch_size{std::numeric_limits<std::size_t>::max()};
    uint64_t    rnd_seed{1238};
    bool        verbose{false};
  };

 public:
  dnnd_kernel(const option& opt, const point_store_type& point_store,
              const point_partitioner&      partitioner,
              const distance_function_type& distance_function, ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_function(distance_function),
        m_comm(comm),
        m_rnd_generator(m_option.rnd_seed + m_comm.rank()) {
    priv_find_max_id();
    m_this.check(m_comm);
  }

  /// \brief Construct a knn-index.
  /// \param knn_index k-nn index instance to store the constructed one.
  template <typename index_alloc_type>
  void construct(
      nn_index<id_type, distance_type, index_alloc_type>& knn_index) {
    if (m_option.verbose) {
      m_comm.cout0() << "Running NN-Descent kernel" << std::endl;
    }
    priv_init_knn_heap_with_random_values();
    priv_construct_kernel();
    priv_convert(knn_index);
  }

  /// \brief Construct a knn-index, starting from a given initial neighbors.
  /// \param init_knn_index Initial neighbors. The distance values will not
  /// be used.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param knn_index k-nn index instance to store the constructed one.
  template <typename init_index_alloc_type, typename index_alloc_type>
  void construct(
      const nn_index<id_type, distance_type, init_index_alloc_type>&
                                                          init_knn_index,
      const bool                                          recheck,
      nn_index<id_type, distance_type, index_alloc_type>& knn_index) {
    if (m_option.verbose) {
      m_comm.cout0() << "Running NN-Descent kernel" << std::endl;
    }
    priv_init_knn_heap_with_index(init_knn_index, recheck);
    priv_construct_kernel();
    priv_convert(knn_index);
  }

  /// \brief Construct a knn-index, starting from a given initial neighbors.
  /// \param init_knn_index Initial neighbors.
  /// \param recheck If true, redo the neighbor check for the initial index,
  /// i.e., mark the initial neighbors as 'new' neighbors.
  /// \param knn_index k-nn index instance to store the constructed one.
  template <typename alloc_type>
  void construct(
      const std::unordered_map<id_type, std::vector<id_type>>& init_knn_index,
      const bool                                               recheck,
      nn_index<id_type, distance_type, alloc_type>&            knn_index) {
    if (m_option.verbose) {
      m_comm.cout0() << "Running NN-Descent kernel" << std::endl;
    }
    priv_init_knn_heap_with_index(init_knn_index, recheck);
    priv_construct_kernel();
    priv_convert(knn_index);
  }

  ygm::comm& comm() { return m_comm; }

 private:
  using self_type = dnnd_kernel<PointStore, Distance>;

  // bool is the flag to represent if the neighbor has been selected for the
  // friend checking.
#if SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
  using knn_heap_table_type =
      boost::unordered_node_map<id_type,
                                unique_knn_heap<id_type, distance_type, bool>>;
#else
  using knn_heap_table_type =
      std::unordered_map<id_type,
                         unique_knn_heap<id_type, distance_type, bool>>;
#endif

  using neighbor_type = neighbor<id_type, distance_type>;
#if SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
  using adj_lsit_type =
      boost::unordered_node_map<id_type, std::vector<id_type>>;
#else
  using adj_lsit_type = std::unordered_map<id_type, std::vector<id_type>>;
#endif

  static constexpr std::size_t k_neighbor_check_local_batch_size_factor = 4;

  void priv_find_max_id() {
    m_global_max_id = 0;
    for (const auto& [id, _] : m_point_store) {
      m_global_max_id = std::max(m_global_max_id, id);
    }
    m_global_max_id = m_comm.all_reduce_max(m_global_max_id);
  }

  void priv_init_knn_heap_with_random_values() {
    if (m_option.verbose) {
      m_comm.cout0() << "\nInitializing the k-NN index with random neighbors."
                     << std::endl;
    }
    priv_allocate_knn_heap();
    priv_fill_knn_heap_with_random_value();
  }

  template <typename init_index_store_type>
  void priv_init_knn_heap_with_index(
      const init_index_store_type& init_knn_index, const bool recheck) {
    if (m_option.verbose) {
      m_comm.cout0()
          << "\nInitializing the k-NN index using the given initial neighbors."
          << std::endl;
    }
    priv_allocate_knn_heap();
    priv_fill_knn_heap_with_initial_index(init_knn_index, recheck);
    // Fill the remaining uninitialized space with random values
    priv_fill_knn_heap_with_random_value();
  }

  void priv_construct_kernel() {
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
    m_num_neighbor_suggestion_msgs = 0;
    m_num_feature_msgs             = 0;
    m_num_distance_msgs            = 0;
    m_num_pruned_distance_msgs     = 0;
#endif
    std::size_t epoch_no = 0;
    while (true) {
      if (m_option.verbose) {
        m_comm.cout0() << "\n[Epoch\t" << epoch_no << "]" << std::endl;
      }
      ygm::timer epoch_timer;

      ygm::timer    gen_timer;
      adj_lsit_type old_table;
      adj_lsit_type new_table;
      priv_get_old_and_new(old_table, new_table);
      m_comm.cf_barrier();
      if (m_option.verbose) {
        m_comm.cout0() << "Generating friend checking requests took (s)\t"
                       << gen_timer.elapsed() << std::endl;
      }

      m_cnt_new_neighbors = 0;
      priv_update_neighbors(old_table, new_table);
      m_comm.cf_barrier();

      if (m_option.verbose) {
        m_comm.cout0() << "\nEpoch took (s)\t" << epoch_timer.elapsed()
                       << std::endl;
      }
      // Test the terminal condition
      const auto num_global_news = m_comm.all_reduce_sum(m_cnt_new_neighbors);
      if (m_option.verbose) {
        m_comm.cout0() << "#of neighbor updates\t" << num_global_news
                       << std::endl;
      }
      if ((double)num_global_news <
          m_option.delta * (m_global_max_id + 1) * m_option.k) {
        break;
      }
      ++epoch_no;
    }
    m_comm.cf_barrier();
    if (m_option.verbose) {
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
      m_comm.cout0() << "\nMessage Statistics" << std::endl;
      m_comm.cout0() << "#of sent neighbor suggestions\t"
                     << m_comm.all_reduce_sum(m_num_neighbor_suggestion_msgs)
                     << std::endl;
      m_comm.cout0() << "#of sent feature vectors\t"
                     << m_comm.all_reduce_sum(m_num_feature_msgs) << std::endl;
      m_comm.cout0() << "#of returned distance\t"
                     << m_comm.all_reduce_sum(m_num_distance_msgs) << std::endl;
      m_comm.cout0() << "#of pruned messages due to longer distance\t"
                     << m_comm.all_reduce_sum(m_num_pruned_distance_msgs)
                     << std::endl;
#endif
    }

#if SALTATLAS_DNND_PROFILE_FEATURE_MSG
    priv_dump_feature_msg_profile();
#endif
  }

  /// \brief Fill k-NN heap with random values.
  /// This function can accept already partially filled heap.
  void priv_fill_knn_heap_with_random_value() {
    m_comm.cf_barrier();
    ygm::timer init_timer;

    // sqrt(k) is enough?
    const std::size_t init_k = m_option.k;

    // Initialize the k-nn heap with random values using a batched algorithm to
    // avoid sending too many messages at once. A single task corresponds to all
    // works of a single point in the dataset to simplify the implementation.
    // Thus, the total number of tasks is equal to the number of points in the
    // dataset. The global batch size is equal to the mini-batch size divided by
    // init_k as each point sends up to init_k messages for initialization.
    auto pitr = m_point_store.begin();
    run_batched_ygm_async(
        m_point_store.size(),               // #of tasks in local
        m_option.mini_batch_size / init_k,  // global batch size
        m_option.verbose, m_comm, [this, &pitr, init_k](auto& comm) {
          const auto  sid          = pitr->first;
          const auto& source_point = pitr->second;

          std::unordered_set<id_type> neighbors;
          // Get the neighbors already in the heap
          if (m_knn_heap_table.count(sid) > 0) {
            for (auto nitr = m_knn_heap_table.at(sid).begin(),
                      nend = m_knn_heap_table.at(sid).end();
                 nitr != nend; ++nitr) {
              const auto& nid = nitr->first;
              neighbors.insert(nid);
            }
          }

          // Fill the remaining space with random values
          while (neighbors.size() < init_k) {
            id_type nid;
            // Generate a random id that is not in the neighbor set
            while (true) {
              std::uniform_int_distribution<id_type> pid_dist(0,
                                                              m_global_max_id);
              assert(m_option.k < m_global_max_id);
              nid = pid_dist(m_rnd_generator);
              if (nid != sid && neighbors.count(nid) == 0) break;
            }

            neighbors.insert(nid);
            // Visit 'nid' and come back to 'sid' with the distance between
            // them.
            m_comm.async(m_point_partitioner(nid), distance_calculator{},
                         m_this, sid, nid, source_point);
          }

          ++pitr;
          return 1;
        });

    if (m_option.verbose) {
      m_comm.cout0() << "Filling initial index took (s)\t"
                     << init_timer.elapsed() << std::endl;
      m_comm.cout0() << "#of generated initial neighbors: "
                     << m_comm.all_reduce_sum(m_knn_heap_table.size() * init_k)
                     << std::endl;
    }
  }

  /// \brief Fills k-NN heap with a given index.
  template <typename alloc>
  void priv_fill_knn_heap_with_initial_index(
      const nn_index<id_type, distance_type, alloc>& init_knn_index,
      const bool                                     recheck) {
    for (auto pitr = init_knn_index.points_begin();
         pitr != init_knn_index.points_end(); ++pitr) {
      const auto& sid = pitr->first;
      for (auto nitr = init_knn_index.neighbors_begin(sid);
           nitr != init_knn_index.neighbors_end(sid); ++nitr) {
        const auto& nid   = nitr->id;
        const auto& point = m_point_store[sid];
        m_comm.async(m_point_partitioner(nid), distance_calculator{}, m_this,
                     sid, nid, point);
      }
    }
    if (!recheck) {
      priv_make_knn_heap_old();
    }
    m_comm.barrier();
  }

  /// \brief Fills k-NN heap with a given index.
  void priv_fill_knn_heap_with_initial_index(
      const std::unordered_map<id_type, std::vector<id_type>>& init_knn_index,
      const bool                                               recheck) {
    for (auto pitr = init_knn_index.begin(); pitr != init_knn_index.end();
         ++pitr) {
      const auto& sid = pitr->first;
      for (auto nitr = pitr->second.begin(); nitr != pitr->second.end();
           ++nitr) {
        const auto& nid   = *nitr;
        const auto& point = m_point_store[sid];
        m_comm.async(m_point_partitioner(nid), distance_calculator{}, m_this,
                     sid, nid, point);
      }
    }
    if (!recheck) {
      priv_make_knn_heap_old();
    }
    m_comm.barrier();
  }

  void priv_make_knn_heap_old() {
    for (auto& item : m_knn_heap_table) {
      auto& heap = item.second;
      for (auto nitr = heap.begin(), nend = heap.end(); nitr != nend; ++nitr) {
        const auto& id = nitr->first;
        heap.value(id) = false;
      }
    }
  }

  void priv_allocate_knn_heap() {
    m_comm.cf_barrier();
    ygm::timer timer;

    m_knn_heap_table.clear();

    m_knn_heap_table.reserve(m_point_store.size());
    for (auto itr = m_point_store.begin(); itr != m_point_store.end(); ++itr) {
      const auto sid = itr->first;
      m_knn_heap_table.emplace(sid, m_option.k);
    }

    m_comm.cf_barrier();
    if (m_option.verbose) {
      m_comm.cout0() << "Allocating k-NN heap took (s)\t" << timer.elapsed()
                     << std::endl;
    }
  }

  struct distance_calculator {
    // Calculate the distance between sid and nid on the rank nid is assigned.
    // Then, send back the calculated distance value to sid.
    void operator()(const ygm::ygm_ptr<self_type>& local_this,
                    const id_type sid, const id_type nid,
                    const point_type& src_point) {
      const auto& nbr_point = local_this->m_point_store[nid];
      const auto  d = local_this->m_distance_function(src_point, nbr_point);
      local_this->comm().async(local_this->m_point_partitioner(sid),
                               distance_calculator{}, local_this, sid, nid, d);
    }

    // Push a returned distance to sid's heap.
    void operator()(ygm::ygm_ptr<self_type> local_this, const id_type sid,
                    const id_type nid, const distance_type d) {
      assert(local_this->m_knn_heap_table.count(sid));
      local_this->m_knn_heap_table.at(sid).push_unique(nid, d, true);
    }
  };

  void priv_get_old_and_new(adj_lsit_type& old_table,
                            adj_lsit_type& new_table) {
    priv_select_outgoing_old_and_new(old_table, new_table);
    priv_add_reverse_neighbors(old_table);
    priv_add_reverse_neighbors(new_table);
  }

  void priv_select_outgoing_old_and_new(adj_lsit_type& old_table,
                                        adj_lsit_type& new_table) {
    old_table.clear();
    new_table.clear();
    old_table.reserve(m_knn_heap_table.size());
    new_table.reserve(m_knn_heap_table.size());

    for (auto sitr = m_knn_heap_table.begin(), end = m_knn_heap_table.end();
         sitr != end; ++sitr) {
      const auto& source = sitr->first;
      auto&       olds   = old_table[source];
      auto&       news   = new_table[source];
      olds.clear();
      news.clear();

      auto& neighbors = sitr->second;
      for (auto nitr = neighbors.begin(), nend = neighbors.end(); nitr != nend;
           ++nitr) {
        const auto& nid          = nitr->first;
        const bool  new_neighbor = nitr->second;
        if (new_neighbor) {
          news.push_back(nid);
        } else {
          olds.push_back(nid);
        }
      }

      // Select 'new' items to use for the friend checking.
      const int num_news =
          std::min((int)(m_option.r * m_option.k), (int)news.size());
      std::shuffle(news.begin(), news.end(), m_rnd_generator);
      news.erase(news.begin() + num_news, news.end());
      assert(news.size() == num_news);
      // reset their 'new' flags as they have been selected.
      for (const auto& nid : news) {
        neighbors.value(nid) = false;
      }
    }
  }

  void priv_add_reverse_neighbors(adj_lsit_type& table) {
    auto r_table = priv_gen_reversed_table(table);
    if (m_option.exchange_reverse_neighbors) {
      r_table = priv_exchange_reverse_neighbors(r_table);
    }
    priv_merge_reversed_table(r_table, table);

    priv_remove_duplicates(table);
  }

  adj_lsit_type priv_gen_reversed_table(const adj_lsit_type& table) {
    adj_lsit_type r_table;
    for (const auto& elem : table) {
      const auto& source    = elem.first;
      const auto& neighbors = elem.second;
      for (const auto& n : neighbors) {
        r_table[n].push_back(source);
      }
    }
    return r_table;
  }

  /// Note: this function destroy the contents of reversed_table.
  void priv_merge_reversed_table(adj_lsit_type& reversed_table,
                                 adj_lsit_type& table) {
    for (auto& [source, r_neighbors] : reversed_table) {
      const auto num_items_to_keep =
          std::min((std::size_t)(m_option.r * m_option.k), r_neighbors.size());
      std::shuffle(r_neighbors.begin(), r_neighbors.end(), m_rnd_generator);
      r_neighbors.erase(r_neighbors.begin() + num_items_to_keep,
                        r_neighbors.end());
      assert(r_neighbors.size() == num_items_to_keep);

      table[source].insert(table[source].end(), r_neighbors.begin(),
                           r_neighbors.end());
      r_neighbors.clear();
      r_neighbors.shrink_to_fit();
    }
    reversed_table.clear();
    reversed_table.rehash(0);
  }

  static void priv_remove_duplicates(adj_lsit_type& table) {
    for (auto& item : table) {
      auto& neighbors = item.second;
      std::sort(neighbors.begin(), neighbors.end());
      const auto last = std::unique(neighbors.begin(), neighbors.end());
      neighbors.erase(last, neighbors.end());
    }
  }

  adj_lsit_type priv_exchange_reverse_neighbors(
      const adj_lsit_type& reverse_neighbors) {
    // Only one call is allowed at a time within a process.
    static std::mutex           mutex;
    std::lock_guard<std::mutex> guard(mutex);

    // Randomize the source ID's order to avoid updating the same node from
    // many processes at the same time.
    std::vector<id_type> source_ids;
    std::size_t          num_neighbors_to_send = 0;
    {
      source_ids.reserve(reverse_neighbors.size());
      for (const auto& rns : reverse_neighbors) {
        source_ids.push_back(rns.first);
        num_neighbors_to_send += rns.second.size();
      }
      std::shuffle(source_ids.begin(), source_ids.end(), m_rnd_generator);
    }

    // Exchange the number of neighbors to send.
#if SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
    static boost::unordered_flat_map<id_type, std::size_t> count_incoming;
#else
    static std::unordered_map<id_type, std::size_t> count_incoming;
#endif
    {
      count_incoming.reserve(m_point_store.size());
      for (auto& item : m_point_store) {
        count_incoming[item.first] = 0;
      }
      m_comm.cf_barrier();

      auto itr = reverse_neighbors.begin();
      run_batched_ygm_async(
          reverse_neighbors.size(), m_option.mini_batch_size, false, m_comm,
          [&itr, this](auto& comm) {
            const auto& source        = itr->first;
            const auto  num_neighbors = itr->second.size();
            comm.async(
                m_point_partitioner(source),
                [](const std::size_t id, const std::size_t n) {
                  count_incoming[id] += n;
                },
                source, num_neighbors);
            ++itr;
            return 1;
          });
      assert(itr == reverse_neighbors.end());
    }

    // Init the receive buffer.
    adj_lsit_type         recv_buf;
    static adj_lsit_type& ref_recv_buf = recv_buf;
    {
      ref_recv_buf.clear();
      ref_recv_buf.reserve(count_incoming.size());
      for (auto& item : count_incoming) {
        ref_recv_buf[item.first].reserve(item.second);
      }
      count_incoming.clear();
      count_incoming.rehash(0);
      m_comm.cf_barrier();
    }

    // Send reverse neighbors to the corresponding sources.
    {
      auto sitr            = source_ids.begin();
      auto neighbor_sender = [&sitr, &reverse_neighbors,
                              this](ygm::comm& comm) {
        const auto& src = *sitr;
        const auto& rn  = reverse_neighbors.at(src);
        for (const auto& n : rn) {
          comm.async(
              m_point_partitioner(src),
              [](const id_type vid, const auto& neighbor) {
                ref_recv_buf[vid].push_back(neighbor);
              },
              src, n);
        }
        ++sitr;
        return 1;
      };

      run_batched_ygm_async(source_ids.size(), m_option.mini_batch_size, false,
                            m_comm, neighbor_sender);
      assert(sitr == source_ids.end());
    }

    return recv_buf;
  }

  void priv_update_neighbors(adj_lsit_type& old_table,
                             adj_lsit_type& new_table) {
    // Shuffle table elements to avoid sending messages to the same process
    // from many ranks at a time.
    std::vector<id_type> new_msg_srcs;
    for (const auto& item : new_table) {
      new_msg_srcs.push_back(item.first);
    }
    std::shuffle(new_msg_srcs.begin(), new_msg_srcs.end(), m_rnd_generator);

    for (auto& item : old_table) {
      std::shuffle(item.second.begin(), item.second.end(), m_rnd_generator);
    }
    for (auto& item : new_table) {
      std::shuffle(item.second.begin(), item.second.end(), m_rnd_generator);
    }
    m_mini_batch_no = 0;
    m_comm.cf_barrier();

    priv_update_neighbors_new_new(new_msg_srcs, new_table);
    priv_update_neighbors_old_new(new_msg_srcs, old_table, new_table);
  }

  void priv_update_neighbors_new_new(const std::vector<id_type>& srcs,
                                     const adj_lsit_type&        new_table) {
    std::queue<std::pair<id_type, id_type>> targets;
    auto task_generator = [this, &srcs, &new_table, &targets](
                              std::size_t& pos_src, std::size_t& pos1,
                              std::size_t& pos2) {
      static const auto local_batch_size =
          (m_option.mini_batch_size / m_comm.size()) *
          k_neighbor_check_local_batch_size_factor;

      for (; pos_src < srcs.size(); ++pos_src) {
        const auto& news = new_table.at(srcs[pos_src]);
        for (; pos1 < news.size(); ++pos1) {
          for (; pos2 < news.size(); ++pos2) {
            const auto u1 = news[pos1];
            const auto u2 = news[pos2];
            if (u1 >= u2) continue;
            targets.push(std::make_pair(u1, u2));
            // Send some messages before generating everything first
            if (local_batch_size > 0 && targets.size() >= local_batch_size) {
              ++pos2;
              return false;
            }
          }
          pos2 = 0;
        }
        pos1 = 0;
      }
      return true;
    };

    std::size_t pos_src = 0;
    std::size_t pos1    = 0;
    std::size_t pos2    = 0;
    while (true) {
      const bool generated_all_tasks = task_generator(pos_src, pos1, pos2);
      priv_launch_neighbor_checking(targets);
      const bool finished = generated_all_tasks && targets.empty();
      if (m_comm.all_reduce_sum((int)finished) == m_comm.size()) break;
    }
  }

  void priv_update_neighbors_old_new(const std::vector<id_type>& new_srcs,
                                     const adj_lsit_type&        old_table,
                                     const adj_lsit_type&        new_table) {
    static const auto local_batch_size =
        (m_option.mini_batch_size / m_comm.size()) *
        k_neighbor_check_local_batch_size_factor;

    std::queue<std::pair<id_type, id_type>> targets;
    auto task_generator = [this, &new_srcs, &old_table, &new_table, &targets](
                              std::size_t& pos_src, std::size_t& pos_old,
                              std::size_t& pos_new) {
      for (; pos_src < new_srcs.size(); ++pos_src) {
        const auto& src  = new_srcs[pos_src];
        const auto& news = new_table.at(src);
        if (old_table.count(src) == 0) continue;
        const auto& olds = old_table.at(src);
        for (; pos_new < news.size(); ++pos_new) {
          for (; pos_old < olds.size(); ++pos_old) {
            const auto u1 = news[pos_new];
            const auto u2 = olds[pos_old];
            if (u1 == u2) continue;
            targets.push(std::make_pair(u1, u2));
            // Send some messages before generating everything first
            if (local_batch_size > 0 && targets.size() >= local_batch_size) {
              ++pos_old;
              return false;
            }
          }
          pos_old = 0;
        }
        pos_new = 0;
      }
      return true;
    };

    std::size_t pos_src = 0;
    std::size_t pos_old = 0;
    std::size_t pos_new = 0;
    while (true) {
      const bool generated_all_tasks =
          task_generator(pos_src, pos_old, pos_new);
      priv_launch_neighbor_checking(targets);
      const bool finished = generated_all_tasks && targets.empty();
      if (m_comm.all_reduce_sum((int)finished) == m_comm.size()) break;
    }
  }

  struct neighbor_updater {
    // Called first.
    // sends u1's point data to u2.
    void operator()(const ygm::ygm_ptr<self_type>& local_this, const id_type u1,
                    const id_type u2) {
      const auto heap = local_this->m_knn_heap_table.at(u1);
      // If the candidate is in the heap,
      // we have already checked the distance between myself and u2.
      // Thus, we don't need to send a new message.
      if (heap.contains(u2)) return;

      const auto max_distance = heap.size() < local_this->m_option.k
                                    ? std::numeric_limits<distance_type>::max()
                                    : heap.top().distance;
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
      ++local_this->m_num_feature_msgs;
#endif
      local_this->comm().async(local_this->m_point_partitioner(u2),
                               neighbor_updater{}, local_this, u1, u2,
                               local_this->m_point_store[u1]
#if SALTATLAS_DNND_PRUNE_LONG_DISTANCE_MSGS
                               ,
                               max_distance
#endif
      );
    }

    // 2nd call.
    // Update u2's knn heap and sends the computed distance to u1, if needed.
    void operator()(const ygm::ygm_ptr<self_type>& local_this, const id_type u1,
                    const id_type u2, const point_type& u1_point,
                    const distance_type& u1_max_distance =
                        std::numeric_limits<distance_type>::max()) {
#if SALTATLAS_DNND_PROFILE_FEATURE_MSG
      if (local_this->m_feature_msg_src_count.count(u1) == 0) {
        local_this->m_feature_msg_src_count[u1] = 0;
      }
      ++local_this->m_feature_msg_src_count[u1];
#endif

      auto& nn_heap = local_this->m_knn_heap_table.at(u2);

      // If u1 is already one of the nearest neighbors,
      // there is nothing to do.
      if (nn_heap.contains(u1)) return;

      // Update u2's heap (nearest neighbors list) if 'u1' is closer than the
      // current neighbors.
      const auto& u2_point = local_this->m_point_store[u2];
      const auto  d = local_this->m_distance_function(u1_point, u2_point);
      local_this->m_cnt_new_neighbors += nn_heap.push_unique(u1, d, true);

      if (d < u1_max_distance) {
        local_this->comm().async(local_this->m_point_partitioner(u1),
                                 neighbor_updater{}, local_this, u1, u2, d);
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
        ++local_this->m_num_distance_msgs;
#endif
      } else {
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
        ++local_this->m_num_pruned_distance_msgs;
#endif
      }
    }

    // Third call,
    // Update the nearest neighbors of u1.
    void operator()(const ygm::ygm_ptr<self_type>& local_this, const id_type u1,
                    const id_type u2, const distance_type& d) {
      auto& nn_heap = local_this->m_knn_heap_table.at(u1);
      local_this->m_cnt_new_neighbors += nn_heap.push_unique(u2, d, true);
    }
  };

  void priv_launch_neighbor_checking(
      std::queue<std::pair<id_type, id_type>>& targets) {
    if (m_option.verbose) {
      m_comm.cout0() << "\nMini-batch No. " << m_mini_batch_no << std::endl;
    }
    ygm::timer mini_batch_timer;

    const auto local_mini_batch_size =
        mpi::assign_tasks(targets.size(), m_option.mini_batch_size,
                          m_comm.rank(), m_comm.size(), m_option.verbose);
    assert(local_mini_batch_size <= targets.size());

#if SALTATLAS_DNND_SHOW_MSG_DST_STATISTICS
    std::vector<std::size_t> msg_dst_count(m_comm.size(), 0);
#endif

    for (std::size_t i = 0; i < local_mini_batch_size; ++i) {
      const auto pair = targets.front();
      targets.pop();
#if SALTATLAS_DNND_SHOW_MSG_DST_STATISTICS
      ++msg_dst_count[m_point_partitioner(pair.first)];
      ++msg_dst_count[m_point_partitioner(pair.second)];
#endif
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
      ++m_num_neighbor_suggestion_msgs;
#endif
      m_comm.async(m_point_partitioner(pair.first), neighbor_updater{}, m_this,
                   pair.first, pair.second);
    }
#if SALTATLAS_DNND_SHOW_MSG_DST_STATISTICS
    priv_show_msg_dst_count_statistics(msg_dst_count);
#endif
    m_comm.barrier();
    if (m_option.verbose) {
      m_comm.cout0() << "Mini-batch took (s)\t" << mini_batch_timer.elapsed()
                     << std::endl;
    }
    ++m_mini_batch_no;
  }

  template <typename allocator>
  void priv_convert(nn_index<id_type, distance_type, allocator>& knn_index) {
    knn_index.reset();
    knn_index.reserve(m_knn_heap_table.size());
    for (auto& item : m_knn_heap_table) {
      const auto& src  = item.first;
      auto&       heap = item.second;
      knn_index.reserve_neighbors(src, heap.size());
      while (!heap.empty()) {
        knn_index.insert(src, heap.top());
        heap.pop();
      }
      knn_index.sort_neighbors(src);
    }
    m_knn_heap_table.clear();
    m_comm.cf_barrier();
  }

  void priv_show_msg_dst_count_statistics(
      const std::vector<std::size_t>& local_table) const {
    std::vector<std::size_t> root_table;
    if (m_comm.rank0()) root_table.resize(m_comm.size(), 0);
    m_comm.cf_barrier();

    SALTATLAS_DNND_CHECK_MPI(MPI_Reduce(local_table.data(), root_table.data(),
                                        m_comm.size(), MPI_UNSIGNED_LONG,
                                        MPI_SUM, 0, MPI_COMM_WORLD));

    if (m_comm.rank0()) {
      std::cout
          << "Statistics of the number of messages each process will receive "
             "(ignoring the effectiveness of the pruning techniques)."
          << std::endl;
      const auto sum =
          std::accumulate(root_table.begin(), root_table.end(), std::size_t(0));
      const auto mean = (double)sum / (double)root_table.size();
      std::cout << "#of total messages " << sum << " with " << root_table.size()
                << " workers" << std::endl;
      std::cout << "Max, Mean, Min:\t" << ""
                << *std::max_element(root_table.begin(), root_table.end())
                << ", " << mean << ", "
                << *std::min_element(root_table.begin(), root_table.end())
                << std::endl;
      double x = 0;
      for (const auto n : root_table) x += std::pow(n - mean, 2);
      const auto dv = std::sqrt(x / root_table.size());
      std::cout << "Standard Deviation " << dv << std::endl;

      for (const auto& n : root_table) std::cout << n << " ";
      std::cout << std::endl;
    }
    m_comm.cf_barrier();
  }

#if SALTATLAS_DNND_PROFILE_FEATURE_MSG
  void priv_dump_feature_msg_profile() {
    std::vector<unsigned long> distribution_table(2001, 0);
    for (const auto& item : m_feature_msg_src_count) {
      const auto& src = item.first;
      const auto& cnt = item.second;
      if (cnt < distribution_table.size()) {
        ++distribution_table[cnt];
      } else {
        ++distribution_table.back();
      }
    }

    const auto local_root = *std::min(m_comm.layout().local_ranks().begin(),
                                      m_comm.layout().local_ranks().end());
    // Gather values withing node
    if (m_comm.rank() != local_root) {
      m_comm.async(
          local_root,
          [](const ygm::ygm_ptr<self_type>& local_this, const auto table) {
            for (std::size_t i = 0; i < table.size(); ++i) {
              local_this->m_m_feature_msg_src_count[i] += table[i];
            }
          },
          m_this, m_feature_msg_src_count);
    }
    m_comm.barrier();

    if (m_comm.rank() == local_root) {
      std::string path("./feature_msg_count");
      if (const char* env_p = std::getenv("FCNT_PATH")) {
        path = env_p;
      }
      path += "-" + std::to_string(m_comm.layout().node_id());

      std::ofstream ofs(path);
      for (std::size_t i = 0; i < distribution_table.size(); ++i) {
        ofs << i << "\t" << distribution_table[i] << '\n';
      }
    }
    m_comm.cf_barrier();
    m_comm.cout0() << "Dumped feature message profile" << std::endl;
  }
#endif

  option                        m_option;
  const point_store_type&       m_point_store;
  const point_partitioner       m_point_partitioner;
  const distance_function_type& m_distance_function;
  ygm::comm&                    m_comm;
  std::mt19937                  m_rnd_generator;
  ygm::ygm_ptr<self_type>       m_this{this};
  knn_heap_table_type           m_knn_heap_table{};
  id_type                       m_global_max_id{0};
  std::size_t                   m_mini_batch_no{0};
  std::size_t                   m_cnt_new_neighbors{0};
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
  std::size_t m_num_neighbor_suggestion_msgs{0};
  std::size_t m_num_feature_msgs{0};
  std::size_t m_num_distance_msgs{0};
  std::size_t m_num_pruned_distance_msgs{0};
#endif

#if SALTATLAS_DNND_PROFILE_FEATURE_MSG
#if SALTATLAS_DNND_USE_BOOST_OPEN_ADDRESS_CONTAINER
  boost::unordered_flat_map<id_type, std::size_t> m_feature_msg_src_count;
#else
  boost::unordered_map<id_type, std::size_t> m_feature_msg_src_count;
#endif
#endif
};

}  // namespace saltatlas::dndetail
