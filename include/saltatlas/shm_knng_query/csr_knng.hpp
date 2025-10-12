// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <scoped_allocator>
#include <string>
#include <vector>

#include "saltatlas/dnnd/detail/utilities/omp.hpp"

namespace saltatlas {

template <typename Id, typename Offset = std::size_t,
          typename Alloc = std::allocator<std::byte>>
class csr_knng {
 private:
  using self_type = csr_knng<Id, Offset, Alloc>;
  using byte_allocator_type =
      typename std::allocator_traits<Alloc>::template rebind_alloc<std::byte>;
  using byte_allocator_traits = std::allocator_traits<byte_allocator_type>;
  using byte_pointer          = typename byte_allocator_traits::pointer;

 public:
  using id_type        = Id;
  using offset_type    = Offset;
  using allocator_type = byte_allocator_type;

 private:
  using id_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<id_type>;
  using offset_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<offset_type>;

 public:
  using id_pointer = typename std::allocator_traits<id_allocator_type>::pointer;
  using const_id_pointer =
      typename std::allocator_traits<id_allocator_type>::const_pointer;
  using offset_pointer =
      typename std::allocator_traits<offset_allocator_type>::pointer;
  using const_offset_pointer =
      typename std::allocator_traits<offset_allocator_type>::const_pointer;

  explicit csr_knng(const std::vector<std::filesystem::path>& index_file_paths,
                    const allocator_type& alloc = allocator_type{})
      : m_allocator(alloc) {
    priv_load_index(index_file_paths);
  }

  csr_knng(const csr_knng&)            = delete;
  csr_knng& operator=(const csr_knng&) = delete;

  csr_knng(csr_knng&&)            = default;
  csr_knng& operator=(csr_knng&&) = default;

  ~csr_knng() { priv_deallocate_arrays(); }

  offset_pointer offsets() { return m_offsets; }

  const_offset_pointer offsets() const { return m_offsets; }

  id_pointer neighbors() { return m_neighbors; }

  const_id_pointer neighbors() const { return m_neighbors; }

  std::pair<const id_type*, const id_type*> neighbors(
      const std::size_t pid) const {
    assert(pid < m_num_points);
    return {neighbors_begin(pid), neighbors_end(pid)};
  }

  const id_type* neighbors_begin(const std::size_t pid) const {
    assert(pid < m_num_points);
    return std::to_address(m_neighbors) + m_offsets[pid];
  }

  const id_type* neighbors_end(const std::size_t pid) const {
    assert(pid < m_num_points);
    return std::to_address(m_neighbors) + m_offsets[pid + 1];
  }

  std::size_t num_points() const { return m_num_points; }

  std::size_t num_total_neighbors() const { return m_num_total_neighbors; }

  std::size_t num_neighbors(const std::size_t pid) const {
    assert(pid < m_num_points);
    return m_offsets[pid + 1] - m_offsets[pid];
  }

 private:
  template <typename T>
  static void priv_check_value_range(const T& value, const T& min,
                                     const T& max) {
    if (!(0 <= value && value <= max)) {
      std::cerr << value << " is out of range [ " << min << ", " << max << " ]"
                << std::endl;
      std::abort();
    }
  }

  void priv_allocate_arrays(const std::size_t num_points,
                            const std::size_t num_total_neighbors) {
    assert(!m_buf);
    const std::size_t bytes = (num_points + 1) * sizeof(offset_type) +
                              num_total_neighbors * sizeof(id_type);
    std::cout << "Allocating " << bytes << " bytes..." << std::endl;

    m_buf       = byte_allocator_traits::allocate(m_allocator, bytes);
    m_offsets   = reinterpret_cast<offset_type*>(std::to_address(m_buf));
    m_neighbors = reinterpret_cast<id_type*>(
        std::to_address(m_buf) +
        std::ptrdiff_t((num_points + 1) * sizeof(offset_type)));
    m_num_points          = num_points;
    m_num_total_neighbors = num_total_neighbors;
  }

  void priv_deallocate_arrays() {
    if (!m_buf) return;
    const std::size_t bytes = (m_num_points + 1) * sizeof(offset_type) +
                              m_num_total_neighbors * sizeof(id_type);
    byte_allocator_traits::deallocate(m_allocator, std::to_address(m_buf),
                                      bytes);
    m_buf                 = nullptr;
    m_offsets             = nullptr;
    m_neighbors           = nullptr;
    m_num_points          = 0;
    m_num_total_neighbors = 0;
  }

  /// \warning This function assumes that the index file contains the point IDs
  /// in the first column and distances are not included.
  void priv_load_index(
      const std::vector<std::filesystem::path>& index_file_paths) {
    std::cout << "Loading k-NN index from " << index_file_paths.size()
              << " files..." << std::endl;

    std::size_t num_points          = 0;
    std::size_t num_total_neighbors = 0;

    OMP_DIRECTIVE (parallel for reduction(+ : num_points, num_total_neighbors))
    for (std::size_t i = 0; i < index_file_paths.size(); ++i) {
      const auto&   index_file_path = index_file_paths[i];
      std::ifstream ifs(index_file_path);
      if (!ifs) {
        std::cerr << "Cannot open " << index_file_path << std::endl;
        std::abort();
      }

      std::string buf;
      while (std::getline(ifs, buf)) {
        ++num_points;
        std::istringstream iss(buf);
        id_type            dummy;
        iss >> dummy;  // the first element is the point id
        while (iss >> dummy) {
          ++num_total_neighbors;
        }
      }
    }

    std::cout << "#of points: " << num_points << std::endl;
    std::cout << "#of total neighbors: " << num_total_neighbors << std::endl;
    priv_check_value_range<std::size_t>(
        num_total_neighbors, 0, std::numeric_limits<offset_type>::max());

    priv_allocate_arrays(num_points, num_total_neighbors);

    std::cout << "Constructing offset array" << std::endl;
    OMP_DIRECTIVE (parallel for)
    for (std::size_t i = 0; i < index_file_paths.size(); ++i) {
      const auto&   index_file_path = index_file_paths[i];
      std::ifstream ifs(index_file_path);
      if (!ifs) {
        std::cerr << "Cannot open " << index_file_path << std::endl;
        std::abort();
      }

      std::string buf;
      while (std::getline(ifs, buf)) {
        std::istringstream iss(buf);
        id_type            point_id;
        iss >> point_id;
        priv_check_value_range<uint64_t>(point_id, 0, num_points - 1);

        id_type neighbor_id;
        while (iss >> neighbor_id) {
          priv_check_value_range<uint64_t>(neighbor_id, 0, num_points - 1);
          ++m_offsets[point_id];
        }
      }
    }
    std::partial_sum(m_offsets, m_offsets + num_points, m_offsets);
    assert(m_offsets[num_points - 1] == num_total_neighbors);
    // shift everything to the right by 1
    std::rotate(std::reverse_iterator(m_offsets + num_points + 1),
                std::reverse_iterator(m_offsets + num_points + 1) + 1,
                std::reverse_iterator(m_offsets - 1));
    m_offsets[0] = 0;
    assert(m_offsets[num_points] == num_total_neighbors);

    std::cout << "Constructing neighbor array" << std::endl;
    OMP_DIRECTIVE (parallel for)
    for (std::size_t i = 0; i < index_file_paths.size(); ++i) {
      const auto&   index_file_path = index_file_paths[i];
      std::ifstream ifs(index_file_path);
      if (!ifs) {
        std::cerr << "Cannot open " << index_file_path << std::endl;
        std::abort();
      }

      std::string buf;
      while (std::getline(ifs, buf)) {
        std::istringstream iss(buf);
        id_type            point_id;
        iss >> point_id;
        id_type neighbor_id;
        while (iss >> neighbor_id) {
          m_neighbors[m_offsets[point_id]] = neighbor_id;
          ++m_offsets[point_id];
        }
      }
    }
    // print_arrays();

    // shift everything to the right by 1
    std::rotate(std::reverse_iterator(m_offsets + num_points + 1),
                std::reverse_iterator(m_offsets + num_points + 1) + 1,
                std::reverse_iterator(m_offsets - 1));
    assert(m_offsets[num_points] == num_total_neighbors);
    m_offsets[0] = 0;

    std::cout << "Finished constructing k-NN index" << std::endl;

    // print_arrays();
  }

  void priv_show() const {
    // print offsets
    std::cout << "Offsets: ";
    for (std::size_t i = 0; i < m_num_points + 1; ++i) {
      std::cout << m_offsets[i] << " ";
    }
    std::cout << std::endl;

    // print neighbors
    std::cout << "Neighbors: ";
    for (std::size_t i = 0; i < m_num_total_neighbors; ++i) {
      std::cout << m_neighbors[i] << " ";
    }
    std::cout << std::endl;
  }
  allocator_type m_allocator;
  byte_pointer   m_buf{nullptr};
  offset_pointer m_offsets{nullptr};
  id_pointer     m_neighbors{nullptr};
  std::size_t    m_num_points{0};
  std::size_t    m_num_total_neighbors{0};
};
}  // namespace saltatlas
