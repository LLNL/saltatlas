// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdlib>
#include <utility>

namespace saltatlas::dndetail {

/// \brief Divides a length into multiple groups.
/// \param length A length to be divided.
/// \param block_no A block number.
/// \param num_blocks The number of total blocks.
/// \return The begin and end index of the range. Note that [begin, end).
inline std::pair<std::size_t, std::size_t> partial_range(
    const std::size_t length, const std::size_t block_no,
    const std::size_t num_blocks) {
  std::size_t partial_length = length / num_blocks;
  std::size_t r              = length % num_blocks;

  std::size_t begin_index;

  if (block_no < r) {
    begin_index = (partial_length + 1) * block_no;
    ++partial_length;
  } else {
    begin_index = (partial_length + 1) * r + partial_length * (block_no - r);
  }

  return std::make_pair(begin_index, begin_index + partial_length);
}

}  // namespace saltatlas::dndetail