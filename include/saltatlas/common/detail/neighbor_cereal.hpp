// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cereal/cereal.hpp>

#include <saltatlas/common/detail/neighbor.hpp>

namespace cereal {
/// \brief Save function for sending saltatlas::detail::neighbor using cereal.
template <typename Archive, typename id_type, typename distance_type>
inline void CEREAL_SAVE_FUNCTION_NAME(
    Archive                                                     &archive,
    const saltatlas::detail::neighbor<id_type, distance_type> &data) {
  archive(data.id);
  archive(data.distance);
}

/// \brief Load function for sending saltatlas::detail::neighbor using cereal.
template <typename Archive, typename id_type, typename distance_type>
inline void CEREAL_LOAD_FUNCTION_NAME(
    Archive                                               &archive,
    saltatlas::detail::neighbor<id_type, distance_type> &data) {
  archive(data.id);
  archive(data.distance);
}
}  // namespace cereal