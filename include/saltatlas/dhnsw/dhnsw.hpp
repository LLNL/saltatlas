// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/comm.hpp>

#include <filesystem>
#include <saltatlas/dhnsw/detail/hnswlib_space_wrapper.hpp>
#include <saltatlas/dhnsw/detail/leaf_hnsw.hpp>
#include <saltatlas/dhnsw/detail/local_hnsw.hpp>
#include <saltatlas/dhnsw/detail/logical_physical_mapper.hpp>
#include <saltatlas/dhnsw/detail/metric_hyperplane_partitioner.hpp>
#include <saltatlas/dhnsw/detail/query_manager.hpp>

#include <saltatlas/common/point_store.hpp>

#include <saltatlas/common/data_reader.hpp>
#include <saltatlas/common/detail/neighbor.hpp>
#include <saltatlas/common/detail/utilities/iterator_proxy.hpp>

namespace saltatlas {

struct dhnsw_params {
  uint32_t num_partitions;
  uint32_t max_voronoi_rank         = 16;
  size_t   hnsw_M                   = 16;
  size_t   hnsw_ef_construction     = 200;
  size_t   hnsw_random_seed         = 100;
  uint32_t num_suggested_partitions = 10;
};

template <typename Id, typename Point, typename Distance = double,
          typename Allocator = std::allocator<std::byte>>
class dhnsw {
 public:
  using id_type           = Id;
  using point_type        = Point;
  using distance_type     = Distance;
  using allocator_type    = Allocator;
  using point_store_type  = point_store<id_type, point_type, std::hash<id_type>,
                                        std::equal_to<id_type>, allocator_type>;
  using partition_id_type = uint32_t;

  /// \brief Neighbor type (contains a neighbor ID and the distance to the
  /// neighbor).
  using neighbor_type = typename detail::neighbor<id_type, distance_type>;

  using iterator_proxy_type =
      detail::iterator_proxy<typename point_store_type::const_iterator>;

 private:
  using point_partitioner =
      metric_hyperplane_partitioner<distance_type, id_type, point_type>;

 public:
  template <typename distance_function_type>
  dhnsw(const distance_function_type &distance_func, ygm::comm &comm)
      : dhnsw(distance_func, comm, dhnsw_params(comm.size())) {}

  template <typename distance_function_type>
  dhnsw(const distance_function_type &distance_func, ygm::comm &comm,
        const dhnsw_params &p)
      : m_num_suggested_partitions(p.num_suggested_partitions),
        m_comm(comm),
        m_partitioner(p.num_partitions, m_space_wrapper, m_comm),
        m_mapper(p.num_partitions, comm.size()),
        m_space_wrapper(distance_func),
        m_local_hnsw(m_space_wrapper, m_mapper.local_size(comm.rank()),
                     p.hnsw_M, p.hnsw_ef_construction, p.hnsw_random_seed),
        m_leaf_hnsw(m_space_wrapper, p.num_partitions, p.hnsw_M,
                    p.hnsw_ef_construction, p.hnsw_random_seed) {}

  template <typename id_iterator, typename point_iterator>
  void add_points(id_iterator ids_begin, id_iterator ids_end,
                  point_iterator points_begin, point_iterator points_end) {
    m_point_store.reserve(std::distance(ids_begin, ids_end));
    for (auto id = ids_begin; id != ids_end; ++id) {
      m_point_store[*id] = *points_begin;
      ++points_begin;
    }
  }

  template <typename PathIterator>
  void load_points(PathIterator paths_begin, PathIterator paths_end,
                   const std::string_view file_format) {
    static_assert(
        std::is_same_v<typename std::iterator_traits<PathIterator>::value_type,
                       std::filesystem::path>,
        "paths_iterator must be an iterator of std::filesystem::path");
    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);
    const std::function<int(const id_type &)> partition_identity_function(
        [this](const id_type &id) { return m_comm.rank(); });
    saltatlas::read_points(point_file_paths, file_format, false,
                           partition_identity_function, m_point_store, m_comm);
  }

  template <typename PathIterator>
  void load_points(
      PathIterator paths_begin, PathIterator paths_end,
      const std::function<std::pair<id_type, point_type>(const std::string &)>
          &line_parser) {
    std::vector<std::filesystem::path> point_file_paths(paths_begin, paths_end);

    const auto parser_wrapper = [&line_parser](const std::string &line,
                                               id_type &id, point_type &point) {
      auto ret = line_parser(line);
      id       = ret.first;
      point    = ret.second;
      return true;
    };

    const std::function<int(const id_type &)> partition_identity_function(
        [this](const id_type &id) { return m_comm.rank(); });

    saltatlas::detail::read_points_with_id_helper(
        point_file_paths, parser_wrapper, m_point_store,
        partition_identity_function, m_comm, false);
  }

  void build() {
    m_partitioner.initialize(m_point_store);

    m_comm.cout0("Repartitioning data");
    m_comm.barrier();
    ygm::timer t;
    repartition_data();
    m_comm.barrier();
    m_comm.cout0("Partitioning time: ", t.elapsed());

    m_comm.cout0("Creating HNSW from partitioner leaves");
    m_leaf_hnsw.add_leaf_points(m_partitioner);

    m_comm.cout0("Storing partition recommendations");
    create_partition_recommendations();

    m_comm.cout0("Adding points to HNSWs");
    m_comm.barrier();
    t.reset();
    build_hnsws();
    m_comm.cout0("HNSW construction time: ", t.elapsed());
  };

  template <typename QueryIterator>
  std::vector<std::vector<detail::neighbor<id_type, distance_type>>> query(
      QueryIterator queries_begin, QueryIterator queries_end, const int k) {
    dhnsw_detail::query_manager q(m_comm, m_point_store, m_partitioner,
                                  m_local_hnsw, m_data_partition_recommender,
                                  m_mapper, dhnsw_query_params{k});

    return q.query(queries_begin, queries_end);
  };

  void optimize() {}

  bool contains_local(const id_type id) const {
    return m_point_store.contains(id);
  }

  const point_type &get_local_point(const id_type id) const {
    return m_point_store.at(id);
  }

  auto local_points_begin() const { return m_point_store.begin(); }

  auto local_points_end() const { return m_point_store.end(); }

  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

  std::size_t num_local_points() const { return m_point_store.size(); }

  std::size_t num_points() const {
    return ygm::sum(num_local_points(), m_comm);
  }

 private:
  void /*std::vector<size_t>*/ repartition_data() {
    // Create space for approximately equal splits of data
    size_t global_size = ygm::sum(m_point_store.size(), m_comm);
    static point_store_type s_tmp_point_store;
    s_tmp_point_store.reserve(global_size / m_comm.size());

    // Create vector of sizes for returning. Avoids having to traverse tree in
    // m_partitioner a second time to find partition sizes when creating the
    // local HNSWs and inserting points
    // std::vector<size_t> local_sizes(m_mapper.local_size(m_comm.rank()));
    // ygm::ygm_ptr<std::vector<size_t>> p_local_sizes =
    // m_comm.make_ygm_ptr(local_sizes);

    auto end_iter = m_point_store.end();
    for (auto point_iter = m_point_store.begin(); point_iter != end_iter;
         ++point_iter) {
      const auto &[id, point] = *point_iter;
      auto partition          = m_partitioner.find_partition(point);
      auto local_partition_id = m_mapper.logical_to_local(partition);

      int dest_rank = partition % m_comm.size();

      m_comm.async(
          dest_rank,
          [](const id_type &id, const point_type &point,
             const dhnsw_detail::logical_physical_mapper::local_id_type
                                               local_partition_id/*,
             ygm::ygm_ptr<std::vector<size_t>> p_local_sizes*/) {
            s_tmp_point_store[id] = point;
            //++(*p_local_sizes)[local_partition_id];
          },
          id, point, local_partition_id /*, p_local_sizes*/);
    }

    m_comm.barrier();

    m_point_store = std::move(s_tmp_point_store);
    s_tmp_point_store.reset();

    return /*local_sizes*/;
  }

  void build_hnsws() {
    std::vector<std::vector<id_type>> partitioned_ids(
        m_mapper.local_size(m_comm.rank()));
    std::vector<size_t> local_hnsw_sizes(partitioned_ids.size());

    auto end_iter = m_point_store.end();
    for (auto point_iter = m_point_store.begin(); point_iter != end_iter;
         ++point_iter) {
      const auto &[id, point] = *point_iter;
      auto partition          = m_partitioner.find_partition(point);
      auto local_partition_id = m_mapper.logical_to_local(partition);

      partitioned_ids[local_partition_id].push_back(id);
      ++local_hnsw_sizes[local_partition_id];
    }

    m_local_hnsw.resize(local_hnsw_sizes);

    for (int local_partition_id = 0;
         local_partition_id < partitioned_ids.size(); ++local_partition_id) {
      const auto &curr_partition_ids = partitioned_ids[local_partition_id];
      for (int i = 0; i < curr_partition_ids.size(); ++i) {
        const auto &id = curr_partition_ids[i];
        m_local_hnsw.add_point(m_point_store[id], id, local_partition_id);
      }
    }
  }

  void create_partition_recommendations() {
    auto end_iter = m_point_store.end();
    for (auto point_iter = m_point_store.begin(); point_iter != end_iter;
         ++point_iter) {
      const auto &[id, point] = *point_iter;

      auto point_partition = m_partitioner.find_partition(point);

      std::priority_queue<std::pair<distance_type, size_t>> nearby_leaf_points =
          m_leaf_hnsw.search(point, m_num_suggested_partitions + 1);

      while (not nearby_leaf_points.empty()) {
        const auto &[dist, partition_id] = nearby_leaf_points.top();
        assert(partition_id < m_partitioner.num_partitions());
        if (partition_id != point_partition) {
          m_data_partition_recommender[id].push_back(partition_id);
        }

        nearby_leaf_points.pop();
      }
    }
  }

  ygm::comm &m_comm;

  dhnsw_detail::SpaceWrapper<distance_type, point_type> m_space_wrapper;

  point_partitioner                     m_partitioner;
  dhnsw_detail::logical_physical_mapper m_mapper;

  // dhnsw_detail::dhnsw_impl<distance_type, id_type, point_type,
  // point_partitioner>
  // m_index_impl;
  // dhnsw_detail::query_engine_impl<distance_type, id_type, point_type,
  // point_partitioner>
  // m_query_engine_impl;

  point_store_type m_point_store;

  dhnsw_detail::local_hnsw<distance_type, point_type, id_type> m_local_hnsw;
  dhnsw_detail::leaf_hnsw<distance_type, point_type>           m_leaf_hnsw;

  partition_id_type m_num_suggested_partitions;
  std::unordered_map<id_type, std::vector<partition_id_type>>
      m_data_partition_recommender;
};
}  // namespace saltatlas
