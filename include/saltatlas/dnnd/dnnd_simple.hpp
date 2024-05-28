// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SALTATLAS_INCLUDE_SALTATLAS_DNND_DNND_SIMPLE_HPP_
#define SALTATLAS_INCLUDE_SALTATLAS_DNND_DNND_SIMPLE_HPP_

#include <filesystem>
#include <string_view>

#include <ygm/comm.hpp>

#include "saltatlas/dnnd/data_reader.hpp"
#include "saltatlas/dnnd/detail/base_dnnd.hpp"
#include "saltatlas/dnnd/detail/utilities/iterator_proxy.hpp"
#include "saltatlas/dnnd/feature_vector.hpp"

namespace saltatlas {

/// \brief Distributed NNDescent simple version.
/// \tparam Id Point ID type.
/// \tparam Point Point type.
/// \tparam Distance Distance type.
template <typename Id       = uint64_t,
          typename Point    = saltatlas::feature_vector<double>,
          typename Distance = double>
class dnnd : public dndetail::base_dnnd<Id, Point, Distance> {
 private:
  using base_type      = dndetail::base_dnnd<Id, Point, Distance>;
  using data_core_type = typename base_type::data_core_type;

 public:
  using id_type                = typename base_type::id_type;
  using distance_type          = typename base_type::distance_type;
  using point_store_type       = typename base_type::point_store_type;
  using knn_index_type         = typename base_type::knn_index_type;
  using point_type             = typename base_type::point_type;
  using neighbor_type          = typename base_type::neighbor_type;
  using query_store_type       = typename base_type::query_store_type;
  using point_partitioner      = typename base_type::point_partitioner;
  using neighbor_store_type    = typename base_type::neighbor_store_type;
  using distance_function_type = typename base_type::distance_function_type;
  using iterator_proxy_type =
      dndetail::iterator_proxy<typename point_store_type::const_iterator>;

 private:
  using nn_kernel_type = dndetail::dnnd_kernel<point_store_type, distance_type>;
  using query_kernel_type =
      dndetail::dknn_batch_query_kernel<point_store_type, knn_index_type>;
  using nn_index_optimizer_type =
      dndetail::nn_index_optimizer<point_store_type, knn_index_type>;

 public:
  /// \brief Constructor.
  /// \param did Distance function id.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance::id& did, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : base_type(verbose, comm), m_data_core(distance::id::custom, rnd_seed) {
    m_data_core.distance_id = did;
    base_type::init_data_core(m_data_core);
  }

  /// \brief Constructor.
  /// \param distance_func Distance function.
  /// \param comm YGM comm instance.
  /// \param rnd_seed Seed for random generators.
  /// \param verbose If true, enable the verbose mode.
  dnnd(const distance_function_type& distance_func, ygm::comm& comm,
       const uint64_t rnd_seed = std::random_device{}(),
       const bool     verbose  = false)
      : base_type(verbose, comm), m_data_core(distance::id::custom, rnd_seed) {
    base_type::init_data_core(m_data_core, distance_func);
  }

  template <typename id_iterator, typename point_iterator>
  void add_points(id_iterator ids_begin, id_iterator ids_end,
                  point_iterator points_begin, point_iterator points_end) {
    point_store_type& pstore = base_type::get_point_store();
    pstore.reserve(std::distance(ids_begin, ids_end));
    for (auto id = ids_begin; id != ids_end; ++id) {
      pstore[*id] = *points_begin;
      ++points_begin;
    }
  }

  template <typename paths_iterator>
  void load_points(paths_iterator paths_begin, paths_iterator paths_end,
                   const std::string_view file_format) {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<paths_iterator>::value_type,
            std::filesystem::path>,
        "paths_iterator must be an iterator of std::filesystem::path");
    std::vector<std::string> point_file_paths;
    for (auto path = paths_begin; path != paths_end; ++path) {
      point_file_paths.push_back(path->string());
    }
    saltatlas::read_points(point_file_paths, file_format, false,
                           base_type::get_point_partitioner(),
                           base_type::get_point_store(), base_type::get_comm());
  }

  template <typename paths_iterator>
  void load_points(
      paths_iterator paths_begin, paths_iterator paths_end,
      const std::function<std::pair<id_type, point_type>(const std::string&)>&
          line_parser) {
    std::vector<std::string> point_file_paths(paths_begin, paths_end);

    const auto parser_wrapper = [&line_parser](const std::string& line,
                                               id_type& id, point_type& point) {
      auto ret = line_parser(line);
      id       = ret.first;
      point    = ret.second;
      return true;
    };

    saltatlas::dndetail::read_points_with_id_helper(
        point_file_paths, parser_wrapper, base_type::get_point_store(),
        base_type::get_point_partitioner(), base_type::get_comm(), false);
  }

  void build(const int k, const double rho, const double delta) {
    base_type::construct_index(k, rho, delta, false, 1 << 28);
  }

  void optimize(const bool   make_index_undirected     = false,
                const double pruning_degree_multiplier = 1.5) {
    base_type::optimize_index(make_index_undirected, pruning_degree_multiplier,
                              false);
  }

  template <typename query_iterator>
  auto query(query_iterator queries_begin, query_iterator queries_end,
             const int k, const double epsilon) {
    std::vector<point_type> queries(queries_begin, queries_end);
    return base_type::query_batch(queries, k, epsilon, 0.0, 1 << 28);
  }

  void dump_graph(const std::filesystem::path& path,
                  const bool                   dump_distance = false) const {
    base_type::dump_index(path.string(), dump_distance);
  }

  bool contains_local(const id_type id) const {
    return base_type::get_point_store().contains(id);
  }

  const point_type& get_local_point(const id_type id) const {
    return base_type::get_point_store().at(id);
  }

  auto local_points_begin() const {
    return base_type::get_point_store().begin();
  }

  auto local_points_end() const { return base_type::get_point_store().end(); }

  // API for using for_each
  iterator_proxy_type local_points() const {
    return iterator_proxy_type(local_points_begin(), local_points_end());
  }

 private:
  data_core_type m_data_core;
};

}  // namespace saltatlas

#endif  // SALTATLAS_INCLUDE_SALTATLAS_DNND_DNND_SIMPLE_HPP_
