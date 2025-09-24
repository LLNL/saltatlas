// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdlib>
#include <utility>

#if __has_include(<metall/detail/utilities.hpp>)
#include <metall/detail/utilities.hpp>
#else
#warning "Metall is not found. Some utility functions will not be available."
#endif

namespace saltatlas::detail {

#if __has_include(<metall/detail/utilities.hpp>)
using metall::mtlldetail::log2_dynamic;
using metall::mtlldetail::log_cpt;
using metall::mtlldetail::round_down;
using metall::mtlldetail::round_up;
#endif

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

/// \brief Compute min, max. mean, and standard deviation.
/// \param v A vector of values.
template <typename T>
inline std::tuple<T, T, double, double> get_stats(const std::vector<T>& v) {
  assert(!v.empty());

  T min    = std::numeric_limits<T>::max();
  T max    = std::numeric_limits<T>::lowest();
  T sum    = 0.0;
  T sum_sq = 0.0;

  for (const auto& e : v) {
    min = std::min(min, e);
    max = std::max(max, e);
    sum += e;
    sum_sq += e * e;
  }

  const double mean    = double(sum) / v.size();
  const double var     = double(sum_sq) / v.size() - mean * mean;
  const double std_var = std::sqrt(var);

  return {min, max, mean, std_var};
}

/// \brief Generate a round-robin tournament schedule.
/// \param num_players The number of players. Must be even.
/// \param payler_id The player ID. Must be in the range of [0, num_players).
/// \param out_itr An output iterator to store the opponent player IDs. Does not
/// include the player itself.
template <typename OutIterator>
constexpr void gen_round_robin_tournament(const std::size_t num_players,
                                          const std::size_t payler_id,
                                          OutIterator       out_itr) {
  if (num_players <= 1 || num_players % 2 != 0) {
    return;
  }

  // Generate pairs using the round-robin tournament algorithm.
  for (std::size_t round = 0; round < num_players - 1; ++round) {
    for (std::size_t i = 0; i < num_players / 2; ++i) {
      const std::size_t pair1 =
          (i == 0) ? 0 : ((round + i) % (num_players - 1)) + 1;
      const std::size_t pair2 =
          (round - i + num_players - 1) % (num_players - 1) + 1;

      // If one of the players is me, the other player is my opponent in this
      // round.
      if (pair1 == payler_id) {
        *out_itr = pair2;
        ++out_itr;
      } else if (pair2 == payler_id) {
        *out_itr = pair1;
        ++out_itr;
      }
    }
  }
}
}  // namespace saltatlas::detail
