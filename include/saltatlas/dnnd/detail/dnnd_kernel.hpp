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

#include <algorithm>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <type_traits>

#include <ygm/comm.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/dnnd/detail/distance.hpp>
#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/neighbor_cereal.hpp>
#include <saltatlas/dnnd/detail/nn_index.hpp>
#include <saltatlas/dnnd/detail/point_store.hpp>
#include <saltatlas/dnnd/detail/utilities/mpi.hpp>

namespace saltatlas::dndetail {

template <typename PointStore, typename Distance>
class dnnd_kernel {
 public:
  using id_type              = typename PointStore::id_type;
  using distance_type        = Distance;
  using feature_element_type = typename PointStore::feature_element_type;
  // Redefine point store type so that autocompletion works when writing code.
  using point_store_type   = point_store<id_type, feature_element_type,
                                       typename PointStore::allocator_type>;
  using featur_vector_type = typename point_store_type::feature_vector_type;
  using point_partitioner  = std::function<int(const id_type& id)>;
  using distance_metric =
      distance::metric_type<feature_element_type, distance_type>;

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
              const point_partitioner& partitioner,
              const distance_metric& metric, ygm::comm& comm)
      : m_option(opt),
        m_point_store(point_store),
        m_point_partitioner(partitioner),
        m_distance_function(metric),
        m_comm(comm),
        m_rnd_generator(m_option.rnd_seed + m_comm.rank()) {
    m_global_max_id = m_comm.all_reduce_max(m_point_store.max_id());
    m_comm.cf_barrier();
    m_this.check(m_comm);
  }

  template <typename allocator>
  void construct(nn_index<id_type, distance_type, allocator>& knn_index) {
    if (m_option.verbose) {
      m_comm.cout0() << "\nRunning NN-Descent kernel" << std::endl;
    }
    priv_construct();
    priv_convert(knn_index);
  }

  ygm::comm& comm() { return m_comm; }

 private:
  using self_type = dnnd_kernel<PointStore, Distance>;

  // bool is the flag to represent if the neighbor has been selected for the
  // friend checking.
  using knn_heap_table_type =
      std::unordered_map<id_type,
                         unique_knn_heap<id_type, distance_type, bool>>;
  using neighbor_type = neighbor<id_type, distance_type>;
  using adj_lsit_type = std::unordered_map<id_type, std::vector<id_type>>;

  void priv_construct() {
    if (m_option.verbose) {
      m_comm.cout0() << "Initializing the k-NN index." << std::endl;
    }
    priv_init_knn_heap_random();

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

      adj_lsit_type old_table;
      adj_lsit_type new_table;
      priv_select_old_and_new(old_table, new_table);
      m_comm.cf_barrier();

      m_cnt_new_neighbors = 0;
      priv_update_neighbors(old_table, new_table);
      m_comm.cf_barrier();

      if (m_option.verbose) {
        m_comm.cout0() << "\nepoch took (s)\t" << epoch_timer.elapsed()
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
  }

  void priv_init_knn_heap_random() {
    m_knn_heap_table.clear();
    // Allocate memory first
    m_knn_heap_table.reserve(m_point_store.size());
    for (auto itr = m_point_store.begin(); itr != m_point_store.end(); ++itr) {
      const auto sid = itr->first;
      m_knn_heap_table.emplace(sid, m_option.k);
    }
    m_comm.cf_barrier();

    std::uniform_int_distribution<id_type> dist(0, m_global_max_id);
    assert(m_option.k < m_global_max_id);

    for (auto itr = m_point_store.begin(); itr != m_point_store.end(); ++itr) {
      const auto                              sid     = itr->first;
      const auto&                             feature = itr->second;
      const std::vector<feature_element_type> tmp_feature(feature.begin(),
                                                          feature.end());

      std::unordered_set<id_type> unique_table;
      while (unique_table.size() < m_option.k) {  // sqrt(k) is enough?
        id_type nid;
        while (true) {
          nid = dist(m_rnd_generator);
          if (nid != sid && unique_table.count(nid) == 0) break;
        }
        unique_table.insert(nid);
        m_comm.async(m_point_partitioner(nid), distance_calculator{}, m_this,
                     sid, nid, tmp_feature);
      }
    }
    m_comm.barrier();
  }

  struct distance_calculator {
    // Calculate the distance between sid and nid on the rank nid is assigned.
    // Then, send back the calculated distance value to sid.
    void operator()(ygm::ygm_ptr<self_type> local_this, const id_type sid,
                    const id_type                            nid,
                    const std::vector<feature_element_type>& src_feature_vec) {
      const auto& nbr_feature_vec =
          local_this->m_point_store.feature_vector(nid);
      const auto d = local_this->m_distance_function(src_feature_vec.size(),
                                                     src_feature_vec.data(),
                                                     nbr_feature_vec.data());
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

  void priv_select_old_and_new(adj_lsit_type& old_table,
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
      for (auto nitr = neighbors.ids_begin(), nend = neighbors.ids_end();
           nitr != nend; ++nitr) {
        const auto& nid          = nitr->first;
        const bool  new_neighbor = nitr->second;
        if (new_neighbor) {
          news.push_back(nid);
        } else {
          olds.push_back(nid);
        }
      }

      // Select 'new' items to use for the friend checking.
      const int num_news_to_select =
          std::min((int)(m_option.r * m_option.k), (int)news.size());
      std::shuffle(news.begin(), news.end(), m_rnd_generator);
      news.erase(news.begin() + num_news_to_select, news.end());
      assert(news.size() == num_news_to_select);
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
    adj_lsit_type r_table(table.get_allocator());
    for (const auto& elem : table) {
      const auto& source    = elem.first;
      const auto& neighbors = elem.second;
      for (const auto& n : neighbors) {
        r_table[n].push_back(source);
      }
    }
    return r_table;
  }

  void priv_merge_reversed_table(const adj_lsit_type& reversed_table,
                                 adj_lsit_type&       table) {
    for (const auto& item : reversed_table) {
      const auto& source = item.first;
      auto        r_neighbors(item.second);  // Make a copy on purpose

      const auto num_items =
          std::min((std::size_t)(m_option.r * m_option.k), r_neighbors.size());
      std::shuffle(r_neighbors.begin(), r_neighbors.end(), m_rnd_generator);
      r_neighbors.erase(r_neighbors.begin() + num_items, r_neighbors.end());
      assert(r_neighbors.size() == num_items);

      table[source].insert(table[source].end(), r_neighbors.begin(),
                           r_neighbors.end());
    }
  }

  static void priv_remove_duplicates(adj_lsit_type& table) {
    for (auto& item : table) {
      auto& neighbors = item.second;
      std::sort(neighbors.begin(), neighbors.end());
      const auto last = std::unique(neighbors.begin(), neighbors.end());
      neighbors.erase(last, neighbors.end());
    }
  }

  adj_lsit_type priv_exchange_reverse_neighbors(const adj_lsit_type& table) {
    // Only one call is allowed at a time within a process.
    static std::mutex           mutex;
    std::lock_guard<std::mutex> guard(mutex);

    adj_lsit_type         received;
    static adj_lsit_type& ref_received = received;
    ref_received.clear();
    ref_received.reserve(m_point_store.size());
    m_comm.cf_barrier();

    std::vector<id_type> source_table;
    source_table.reserve(table.size());
    for (const auto& item : table) {
      const auto& source = item.first;
      source_table.push_back(source);
    }
    // Randomize the source ID's order to avoid updating the same node from
    // many processes at the same time.
    std::shuffle(source_table.begin(), source_table.end(), m_rnd_generator);

    for (const auto source : source_table) {
      const auto& neighbors = table.at(source);
      m_comm.async(
          m_point_partitioner(source),
          [](auto, const id_type vid,
             const typename adj_lsit_type::mapped_type& neighbors) {
            ref_received[vid].insert(ref_received[vid].end(), neighbors.begin(),
                                     neighbors.end());
          },
          source, neighbors);
    }
    m_comm.barrier();
    return received;
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
      for (; pos_src < srcs.size(); ++pos_src) {
        const auto& news = new_table.at(srcs[pos_src]);
        for (; pos1 < news.size(); ++pos1) {
          for (; pos2 < news.size(); ++pos2) {
            const auto u1 = news[pos1];
            const auto u2 = news[pos2];
            if (u1 >= u2) continue;
            targets.push(std::make_pair(u1, u2));
            // Send some messages before generating everything first
            if (m_option.mini_batch_size > 0 &&
                targets.size() >= m_option.mini_batch_size) {
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
            if (m_option.mini_batch_size > 0 &&
                targets.size() >= m_option.mini_batch_size) {
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
    // sends u1's feature vector to u2.
    void operator()(ygm::ygm_ptr<self_type> local_this, const id_type u1,
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
      std::vector<feature_element_type> f(
          local_this->m_point_store.feature_vector(u1).begin(),
          local_this->m_point_store.feature_vector(u1).end());
      local_this->comm().async(local_this->m_point_partitioner(u2),
                               neighbor_updater{}, local_this, u1, u2, f
#if SALTATLAS_DNND_PRUNE_LONG_DISTANCE_MSGS
                               ,
                               max_distance
#endif
      );
    }

    // 2nd call.
    // Update u2's knn heap and sends the computed distance to u1, if needed.
    void operator()(ygm::ygm_ptr<self_type> local_this, const id_type u1,
                    const id_type                            u2,
                    const std::vector<feature_element_type>& u1_feature,
                    const distance_type&                     u1_max_distance =
                        std::numeric_limits<distance_type>::max()) {
      auto& nn_heap = local_this->m_knn_heap_table.at(u2);

      // If u1 is already one of the nearest neighbors,
      // there is nothing to do.
      if (nn_heap.contains(u1)) return;

      // Update u2's heap (nearest neighbors list) if 'u1' is closer than the
      // current neighbors.
      const auto& u2_feature = local_this->m_point_store.feature_vector(u2);
      const auto  d          = local_this->m_distance_function(
                    u1_feature.size(), u1_feature.data(), u2_feature.data());
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
    void operator()(ygm::ygm_ptr<self_type> local_this, const id_type u1,
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
    ++m_mini_batch_no;

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
  }

  template <typename allocator>
  void priv_convert(nn_index<id_type, distance_type, allocator>& knn_index) {
    knn_index.clear();
    for (auto& item : m_knn_heap_table) {
      const auto&                src  = item.first;
      auto&                      heap = item.second;
      std::vector<neighbor_type> wk;
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
      std::cout << "Max, Mean, Min:\t"
                << "" << *std::max_element(root_table.begin(), root_table.end())
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

  option                  m_option;
  const point_store_type& m_point_store;
  const point_partitioner m_point_partitioner;
  const distance_metric&  m_distance_function;
  ygm::comm&              m_comm;
  std::mt19937            m_rnd_generator;
  ygm::ygm_ptr<self_type> m_this{this};
  knn_heap_table_type     m_knn_heap_table{};
  id_type                 m_global_max_id{0};
  std::size_t             m_mini_batch_no{0};
  std::size_t             m_cnt_new_neighbors{0};
#if SALTATLAS_DNND_SHOW_BASIC_MSG_STATISTICS
  std::size_t m_num_neighbor_suggestion_msgs{0};
  std::size_t m_num_feature_msgs{0};
  std::size_t m_num_distance_msgs{0};
  std::size_t m_num_pruned_distance_msgs{0};
#endif
};

}  // namespace saltatlas::dndetail