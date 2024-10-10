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
  std::cerr << "str_cast is not implemented for this type." << typeid(T).name()
            << std::endl;
  std::abort();
  assert(false);
  return T{};
}

template <>
inline int8_t str_cast<int8_t>(const std::string &input) {
  return static_cast<int8_t>(std::stoi(input.data()));
}

template <>
inline uint8_t str_cast<uint8_t>(const std::string &input) {
  return static_cast<uint8_t>(std::stoul(input.data()));
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

/// \brief Split a string by whitespace and store elements into a vector of T.
/// \tparam T Type to cast.
/// \param input Input string.
/// \return Vector of T.
template <typename T>
inline std::vector<T> str_split(const std::string &input) {
  std::vector<T>     result;
  std::string        token;
  for (std::stringstream ss(input); ss >> token;) {
    result.push_back(str_cast<T>(token));
  }
  return result;
}

/// \brief Split a string into a vector of char.
/// str_split's specialization for char.
/// \param input Input string.
/// \return Vector of chars.
template <>
inline std::vector<char> str_split(const std::string &input) {
  std::vector<char> result(input.begin(), input.end());
  return result;
}

/// \brief Split a string by delimiter and store elements into a vector of T.
/// \tparam T Type to cast.
/// \param input Input string.
/// \param delimiter Delimiter.
/// \return Vector of T.
template <typename T>
inline std::vector<T> str_split(const std::string &input,
                                const char         delimiter) {
  std::vector<T>     result;
  std::string        token;
  std::istringstream token_stream(input);
  while (std::getline(token_stream, token, delimiter)) {
    result.push_back(str_cast<T>(token));
  }
  return result;
}

}  // namespace saltatlas::dndetail