// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/json/src.hpp>

namespace saltatlas::dndetail {

class counter_db {
 public:
  void add(const std::string& name, const double value) {
    auto it = m_db.find(name);
    if (it == m_db.end()) {
      m_db[name] = {1, value};
    } else {
      it->second.count++;
      it->second.total += value;
    }
  }

  void clear() { m_db.clear(); }

  std::size_t get_count(const std::string& name) const {
    auto it = m_db.find(name);
    if (it == m_db.end()) {
      return 0;
    } else {
      return it->second.count;
    }
  }

  double get_total(const std::string& name) const {
    auto it = m_db.find(name);
    if (it == m_db.end()) {
      return 0.0;
    } else {
      return it->second.total;
    }
  }

  double get_average(const std::string& name) const {
    const auto cnt = get_count(name);
    if (cnt == 0) return 0.0;
    return get_total(name) / cnt;
  }

 private:
  struct entry {
    std::size_t count;
    double      total;
  };

  std::unordered_map<std::string, entry> m_db;
};

}  // namespace saltatlas::neo_dnnd