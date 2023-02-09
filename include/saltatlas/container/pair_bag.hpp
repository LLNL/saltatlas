// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>

namespace ygm {
namespace container {

template <typename IndexType, typename Point>
class pair_bag {
 public:
  using value_type = std::pair<IndexType, Point>;
  using self_type  = pair_bag<IndexType, Point>;
  using impl_type  = ygm::container::detail::bag_impl<value_type>;

  pair_bag(ygm::comm &comm) : m_impl(comm) {}

  void async_insert(const value_type &item) { m_impl.async_insert(item); }

  template <typename Function>
  void for_all(Function fn) {
    m_impl.for_all(fn);
  }

  void clear() { m_impl.clear(); }

  size_t size() { return m_impl.size(); }

  void swap(self_type &s) { m_impl.swap(s.m_impl); }

  template <typename Function>
  void local_for_all(Function fn) {
    m_impl.local_for_all(fn);
  }

  ygm::comm &comm() { return m_impl.comm(); }

  void serialize(const std::string &fname) { m_impl.serialize(fname); }
  void deserialize(const std::string &fname) { m_impl.deserialize(fname); }
  std::vector<value_type> gather_to_vector(int dest) {
    return m_impl.gather_to_vector(dest);
  }
  std::vector<value_type> gather_to_vector() {
    return m_impl.gather_to_vector();
  }

 private:
  impl_type m_impl;
};
}  // namespace container
}  // namespace ygm
