// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <saltatlas/common/detail/utilities/string_cast.hpp>
#include <saltatlas/dnnd/detail/utilities/omp.hpp>

namespace saltatlas {

namespace {
namespace omp = saltatlas::utility::omp;
}

template <typename Id, typename T, typename Alloc = std::allocator<T>>
class packed_point_store {
  static_assert(
      std::is_same_v<T, typename std::allocator_traits<Alloc>::value_type>,
      "Alloc::value_type must be the same as T");

 private:
  using pointer_type = typename std::allocator_traits<Alloc>::pointer;

 public:
  using id_type        = Id;
  using value_type     = T;
  using allocator_type = Alloc;

  packed_point_store(const std::vector<std::filesystem::path>& point_file_paths,
                     const std::string_view                    format,
                     const allocator_type& alloc = allocator_type{})
      : m_allocator(alloc) {
    load_point(point_file_paths, format);
  }

  ~packed_point_store() {
    if (m_data) {
      std::allocator_traits<allocator_type>::deallocate(m_allocator, m_data,
                                                        m_k * m_num_points);
      m_data       = nullptr;
      m_num_points = 0;
      m_k          = 0;
    }
  }

  packed_point_store(const packed_point_store&)            = delete;
  packed_point_store& operator=(const packed_point_store&) = delete;

  packed_point_store(packed_point_store&&)            = default;
  packed_point_store& operator=(packed_point_store&&) = default;

  T* buffer() { return m_data; }

  const T* buffer() const { return m_data; }

  T* at(const std::size_t pid) {
    return const_cast<T*>(const_cast<const packed_point_store*>(this)->at(pid));
  }

  const T* at(const std::size_t pid) const {
    assert(pid < m_num_points);
    return std::to_address(m_data) + pid * m_k;
  }

  std::size_t num_dimensions() const { return m_k; }

  std::size_t num_points() const { return m_num_points; }

 private:
  void init(const std::size_t num_points, const std::size_t k) {
    assert(!m_data);
    assert(num_points > 0);
    assert(k > 0);
    const std::size_t total_elements = k * num_points;
    std::cout << "Allocate for " << total_elements << " elements" << std::endl;
    m_data = std::allocator_traits<allocator_type>::allocate(m_allocator,
                                                             total_elements);
    assert(m_data);
    m_num_points = num_points;
    m_k          = k;
  }

  void load_point(const std::vector<std::filesystem::path>& point_file_paths,
                  const std::string_view                    format) {
    if (!((format == "wsv-id") ||
          (format == "wsv" && point_file_paths.size() == 1))) {
      std::cerr << "Unsupported format: " << format << std::endl;
      std::abort();
    }

    std::cout << "Load points from " << point_file_paths.size() << " files"
              << std::endl;

    std::cout << "Counting #of points..." << std::endl;
    std::size_t              k = 0;  // must init with 0 here
    std::vector<std::size_t> num_points(point_file_paths.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
      const auto&   point_file_path = point_file_paths[i];
      std::ifstream ifs(point_file_path);
      if (!ifs) {
        std::cerr << "Cannot open " << point_file_path << std::endl;
        std::abort();
      }

      std::string buf;
      while (std::getline(ifs, buf)) {
        if (i == 0 && k == 0) {  // assume k is initialized with 0
          // count #of dimensions
          k += saltatlas::detail::str_split<value_type>(buf).size();
          if (format == "wsv-id") {
            --k;  // The first column is ID
          }
        }
        ++num_points[i];
      }
    }
    const std::size_t total_num_points =
        std::accumulate(num_points.begin(), num_points.end(), 0ULL);
    std::cout << "#of points: " << total_num_points << std::endl;
    std::cout << "#of dimensions: " << k << std::endl;

    init(total_num_points, k);

    // Construct ID offsets
    std::vector<id_type> id_offsets;
    if (format != "wsv-id") {
      id_offsets.assign(num_points.begin(), num_points.end());
      std::partial_sum(id_offsets.cbegin(), id_offsets.cend(),
                       id_offsets.begin());
      // shift everything to the right by 1
      std::rotate(id_offsets.rbegin(), id_offsets.rbegin() + 1,
                  id_offsets.rend());
      id_offsets[0] = 0;
    }

    std::cout << "Loading points..." << std::endl;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < point_file_paths.size(); ++i) {
      const auto&   point_file_path = point_file_paths[i];
      std::ifstream ifs(point_file_path);
      if (!ifs) {
        std::cerr << "Cannot open " << point_file_path << std::endl;
        std::abort();
      }

      std::string line_buf;
      while (std::getline(ifs, line_buf)) {
        id_type id;
        // Get ID
        {
          std::istringstream iss(line_buf);
          if (format == "wsv-id") {
            iss >> id;
          } else {
            id = id_offsets[i];
            ++id_offsets[i];
          }
        }

        auto points = saltatlas::detail::str_split<value_type>(line_buf);
        if (format == "wsv-id") {
          points.erase(points.begin());
        }
        if (points.size() != m_k) {
          std::cerr << "Unexpected #of dimensions." << std::endl;
          std::cerr << "read dimensions: " << points.size() << std::endl;
          std::cerr << line_buf << std::endl;
          std::abort();
        }
        std::copy(points.begin(), points.end(), at(id));
      }
    }

    std::cout << "Loaded points" << std::endl;
  }

  allocator_type m_allocator;
  std::size_t    m_k{0};
  std::size_t    m_num_points{0};
  pointer_type   m_data{nullptr};
};

}  // namespace saltatlas