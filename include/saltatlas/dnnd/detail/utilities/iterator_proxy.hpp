// Copyright 2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

namespace saltatlas::dndetail {

template <typename iterator_type>
class iterator_proxy {
 public:
  iterator_proxy(iterator_type begin, iterator_type end)
      : m_begin(begin), m_end(end) {}

  iterator_type begin() const { return m_begin; }

  iterator_type end() const { return m_end; }

 private:
  iterator_type m_begin;
  iterator_type m_end;
};

}  // namespace saltatlas::dndetail