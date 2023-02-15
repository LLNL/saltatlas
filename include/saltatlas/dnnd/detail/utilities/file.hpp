#pragma once

#include <fstream>

namespace saltatlas::dndetail {

/// \brief Reads characters from a stream by chunk in stead of once at a time.
/// \tparam char_type Char type.
/// \tparam traits Traits.
/// \param ifs Input file-based.
/// \param buf Pointer to the character array to store the characters to.
/// \param count Number of characters to read.
/// \param chunk_size Read chunk size.
/// \return Returns true on success; otherwise, false.
template <typename char_type, typename traits = std::char_traits<char_type>>
inline bool read_by_chunk(std::basic_ifstream<char_type, traits>& ifs,
                          char_type* const buf, const std::streamsize count,
                          const std::size_t chunk_size = (1ULL << 30)) {
  for (std::size_t off = 0; off < count; off += chunk_size) {
    const auto actual_read_count = std::min(chunk_size, count - off);
    if (!ifs.read(buf + off, actual_read_count)) {
      return false;
    }
  }
  return !!ifs;
}

}  // namespace saltatlas::dndetail