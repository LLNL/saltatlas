// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdlib>
#include <string>

namespace saltatlas::dndetail {
template <typename T>
inline T str_cast(const std::string &) {
  assert(false);
  return T{};
}

template <>
inline int32_t str_cast<int32_t>(const std::string &input) {
  return static_cast<int32_t>(std::stoi(input.data()));
}

template <>
inline uint32_t str_cast<uint32_t>(const std::string &input) {
  return static_cast<uint32_t>(std::stoul(input.data()));
}

template <>
inline int64_t str_cast<int64_t>(const std::string &input) {
  return std::stol(input.data());
}

template <>
inline uint64_t str_cast<uint64_t>(const std::string &input) {
  return std::stoul(input.data());
}

template <>
inline float str_cast<float>(const std::string &input) {
  return std::stof(input.data());
}

template <>
inline double str_cast<double>(const std::string &input) {
  return std::stod(input.data());
}
}  // namespace saltatlas::dndetail