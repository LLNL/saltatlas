// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <saltatlas/common/detail/utilities/float.hpp>

namespace saltatlas::detail {

template <typename Id, typename Distance>
struct neighbor {
  using id_type       = Id;
  using distance_type = Distance;

  neighbor() = default;

  neighbor(const id_type& _id, const distance_type& _distance)
      : id(_id), distance(_distance) {}

  friend bool operator<(const neighbor& lhd, const neighbor& rhd) {
    if (lhd.distance != rhd.distance) return lhd.distance < rhd.distance;
    return lhd.id < rhd.id;
  }

  template <typename T1, typename T2>
  friend std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng);

  id_type       id;
  distance_type distance;
};

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const neighbor<T1, T2>& ng) {
  os << "id = " << ng.id << ", distance = " << ng.distance;
  return os;
}

template <typename Id, typename Distance>
inline bool operator==(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return lhs.id == rhs.id && nearly_equal(lhs.distance, rhs.distance);
}

template <typename Id, typename Distance>
inline bool operator!=(const neighbor<Id, Distance>& lhs,
                       const neighbor<Id, Distance>& rhs) {
  return !(lhs == rhs);
}

}  // namespace saltatlas::detail
