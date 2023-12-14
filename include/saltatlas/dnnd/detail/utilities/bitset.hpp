// Copyright 2023 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>

namespace saltatlas::dndetail {

class dynamic_bitset {
 private:
  using block_type                          = uint64_t;
  static constexpr uint64_t kNumBitsInBlock = sizeof(block_type) * 8ULL;

 public:
  explicit dynamic_bitset(const std::size_t num_bits)
      : m_block(std::make_unique<block_type[]>(num_blocks(num_bits))),
        m_num_bits(num_bits) {
    reset_all();
  }

  bool get(const uint64_t idx) const {
    const block_type mask =
        (0x1ULL << (kNumBitsInBlock - local_index(idx) - 1));
    return (m_block.get()[global_index(idx)] & mask);
  }

  void set(const uint64_t idx) {
    const block_type mask =
        (0x1ULL << (kNumBitsInBlock - local_index(idx) - 1));
    m_block.get()[global_index(idx)] |= mask;
  }

  void reset(const uint64_t idx) {
    const block_type mask =
        (0x1ULL << (kNumBitsInBlock - local_index(idx) - 1));
    m_block.get()[global_index(idx)] &= ~mask;
  }

  void reset_all() {
    std::memset(m_block.get(), 0, num_blocks(m_num_bits) * sizeof(block_type));
  }

  std::size_t size() { return m_num_bits; }

 private:
  static constexpr std::size_t num_blocks(const uint64_t num_bits) {
    return (num_bits + kNumBitsInBlock - 1) / kNumBitsInBlock;
  }

  static constexpr uint64_t global_index(const uint64_t idx) {
    return (idx / kNumBitsInBlock);
  }

  static constexpr uint64_t local_index(const uint64_t idx) {
    return idx & (kNumBitsInBlock - 1);
  }

  std::unique_ptr<block_type[]> m_block;
  std::size_t                   m_num_bits{0};
};

}  // namespace saltatlas::dndetail