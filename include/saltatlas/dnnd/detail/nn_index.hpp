// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#if __has_include(<metall/container/unordered_map.hpp>) \
  && __has_include(<metall/container/vector.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/unordered_map.hpp>
#include <metall/container/vector.hpp>
#else
#include <unordered_map>
#include <vector>
#endif

#include <saltatlas/dnnd/detail/neighbor.hpp>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

namespace saltatlas::dndetail {

namespace {
namespace container =
#if SALTATLAS_DNND_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

// Forward declaration
template <typename IdType = uint64_t, typename DistanceType = double,
          typename Allocator = std::allocator<std::byte>>
class nn_index;

template <typename IdType, typename DistanceType, typename Allocator>
class nn_index {
 public:
  using id_type        = IdType;
  using distance_type  = DistanceType;
  using neighbor_type  = neighbor<id_type, distance_type>;
  using allocator_type = Allocator;

 private:
  using neighbor_list_type =
      container::vector<neighbor_type,
                        other_allocator<allocator_type, neighbor_type>>;
  using point_table_type = container::unordered_map<
      id_type, neighbor_list_type, std::hash<id_type>, std::equal_to<>,
      other_scoped_allocator<allocator_type,
                             std::pair<const id_type, neighbor_list_type>>>;

  using point_iterator          = typename point_table_type::iterator;
  using const_point_iterator    = typename point_table_type::const_iterator;
  using neighbor_iterator       = typename neighbor_list_type::iterator;
  using const_neighbor_iterator = typename neighbor_list_type::const_iterator;

 public:
  explicit nn_index(const allocator_type &allocator = allocator_type{})
      : m_index(allocator) {}

  nn_index(const nn_index &) = default;
  nn_index(nn_index &&)      = default;

  nn_index &operator=(const nn_index &) = default;
  nn_index &operator=(nn_index &&)      = default;

  // allocator-aware copy constructor
  nn_index(const nn_index &other, const allocator_type &allocator)
      : m_index(other.m_index, allocator) {}

  // allocator-aware move constructor
  nn_index(nn_index &&other, const allocator_type &allocator)
      : m_index(std::move(other.m_index), allocator) {}

  void insert(const id_type &source, const neighbor_type &neighbor) {
    m_index[source].push_back(neighbor);
  }

  void sort_neighbors(const id_type &source) {
    std::sort(m_index[source].begin(), m_index[source].end());
  }

  void sort_and_remove_duplicate_neighbors(const id_type &source) {
    sort_neighbors(source);
    m_index[source].erase(
        std::unique(m_index[source].begin(), m_index[source].end()),
        m_index[source].end());
    m_index[source].shrink_to_fit();
  }

  /// \warning The neighbor list must be sorted beforehand.
  void prune_neighbors(const id_type    &source,
                       const std::size_t num_max_neighbors) {
    if (m_index.at(source).size() > num_max_neighbors) {
      m_index[source].resize(num_max_neighbors);
    }
    m_index[source].shrink_to_fit();
  }

  auto points_begin() { return m_index.begin(); }

  auto points_end() { return m_index.end(); }

  auto points_begin() const { return m_index.begin(); }

  auto points_end() const { return m_index.end(); }

  auto neighbors_begin(const id_type &source) {
    return m_index[source].begin();
  }

  auto neighbors_end(const id_type &source) { return m_index[source].end(); }

  auto neighbors_begin(const id_type &source) const {
    assert(m_index.count(source) > 0);
    return m_index.at(source).begin();
  }

  auto neighbors_end(const id_type &source) const {
    assert(m_index.count(source) > 0);
    return m_index.at(source).end();
  }

  void merge(nn_index<IdType, DistanceType, Allocator> &other) {
    for (const auto &[source, neighbors] : other.m_index) {
      for (const auto &neighbor : neighbors) {
        insert(source, neighbor);
      }
    }
  }

  std::size_t num_points() const { return m_index.size(); }

  std::size_t num_neighbors(const id_type &source) const {
    if (m_index.count(source) == 0) return 0;
    return m_index.at(source).size();
  }

  std::size_t count_all_neighbors() const {
    std::size_t num_neighbors = 0;
    for (const auto &[source, neighbors] : m_index) {
      num_neighbors += neighbors.size();
    }
    return num_neighbors;
  }

  /// \brief Clear contents and reduce the storage usage.
  void reset() {
    m_index.clear();
    m_index.rehash(0);
  }

  /// \brief Clear contents and reduce the storage usage.
  void reset_neighbors(const id_type &source) {
    m_index[source].clear();
    m_index[source].shrink_to_fit();
  }

  void reserve(const std::size_t size) { m_index.reserve(size); }

  void reserve_neighbors(const id_type &source, const std::size_t size) {
    m_index[source].reserve(size);
  }

  allocator_type get_allocator() const { return m_index.get_allocator(); }

  bool empty() const { return m_index.empty(); }

  /// \brief Dump the index to a file.
  /// \param filename The file name to dump the index.
  /// \param dump_distance If true, the distance to each neighbor is also
  /// dumped. \return True if the dump is successful. \details For each neighbor
  /// list, the following lines are dumped:
  /// ```
  /// source_id neighbor_id_1 neighbor_id_2 ...
  /// 0.0 distance_1 distance_2 ...
  /// ```
  /// Each item is separated by a space. The first line is the source id and
  /// neighbor ids. The second line is the dummy value and distances to each
  /// neighbor. The dummy value is just a placeholder so that each neighbor id
  /// and distance pair is stored in the same column.
  bool dump(const std::filesystem::path &filename,
            bool                         dump_distance = false) const {
    std::ofstream ofs(filename);
    if (!ofs) {
      std::cerr << "Failed to open the file: " << filename << std::endl;
      return false;
    }

    for (const auto &[source, neighbors] : m_index) {
      ofs << source;
      for (const auto &neighbor : neighbors) {
        ofs << "\t" << neighbor.id;
      }
      ofs << "\n";

      if (!dump_distance) continue;

      ofs << "0.0";  // dummy distance
      for (const auto &neighbor : neighbors) {
        ofs << "\t" << neighbor.distance;
      }
      ofs << "\n";
    }

    ofs.close();
    if (!ofs) {
      std::cerr << "Failed to close the file: " << filename << std::endl;
      return false;
    }
    return true;
  }

  /// \brief Load the index from a dump file.
  /// \param filename The file name to load the index.
  /// The file must be formatted as described in the `dump` method.
  /// \param contains_distance If true, the input file contains distances.
  /// If false, the input file does not contain distances and distance values
  /// will be uninitialized.
  /// \param overwrite If true, the existing index is overwritten.
  /// \return True if the load is successful.
  bool load(const std::filesystem::path &filename,
            const bool contains_distance = false, const bool overwrite = true) {
    if (overwrite) reset();

    std::ifstream ifs(filename);
    if (!ifs) {
      std::cerr << "Failed to open the file: " << filename << std::endl;
      return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      id_type            source;
      iss >> source;
      if (iss.fail()) {
        std::cerr << "Failed to read the source id: " << line << std::endl;
        return false;
      }
      if (m_index.count(source) > 0) {
        std::cerr << "The source id is duplicated: " << source << std::endl;
        return false;
      }

      neighbor_list_type neighbors;
      id_type            neighbor_id;
      while (iss >> neighbor_id) {
        neighbors.emplace_back(neighbor_id, distance_type{});
      }
      m_index[source] = std::move(neighbors);

      if (contains_distance) {
        std::getline(ifs, line);
        std::istringstream iss(line);
        distance_type      distance;
        iss >> distance;  // dummy distance
        std::size_t index = 0;
        while (iss >> distance) {
          m_index[source][index++].distance = distance;
        }
        if (index != m_index[source].size()) {
          std::cerr << "The number of distances does not match the number of "
                       "neighbors"
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

 private:
  point_table_type m_index;
};

}  // namespace saltatlas::dndetail