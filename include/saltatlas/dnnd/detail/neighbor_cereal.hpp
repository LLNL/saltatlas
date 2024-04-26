// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cereal/cereal.hpp>

#include <saltatlas/dnnd/detail/neighbor.hpp>

namespace cereal {
/// \brief Save function for sending saltatlas::dndetail::neighbor using cereal.
template <typename Archive, typename id_type, typename distance_type>
inline void CEREAL_SAVE_FUNCTION_NAME(
    Archive                                                     &archive,
    const saltatlas::dndetail::neighbor<id_type, distance_type> &data) {
  archive(data.id);
  archive(data.distance);
}

/// \brief Load function for sending saltatlas::dndetail::neighbor using cereal.
template <typename Archive, typename id_type, typename distance_type>
inline void CEREAL_LOAD_FUNCTION_NAME(
    Archive                                               &archive,
    saltatlas::dndetail::neighbor<id_type, distance_type> &data) {
  archive(data.id);
  archive(data.distance);
}
}  // namespace cereal