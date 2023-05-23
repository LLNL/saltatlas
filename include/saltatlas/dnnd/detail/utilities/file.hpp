#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <string_view>
#include <algorithm>

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

/// \brief Search file paths recursively.
/// If a directory path is given, it returns all the file paths in the directory
/// and subdirectories. If a file path is given, it returns the file path.
/// \param path Directory or file path.
/// \return Returns a vector of found file paths.
inline std::vector<std::string> find_file_paths(const std::string_view path) {
  std::vector<std::string> paths;
  if (std::filesystem::is_regular_file(std::filesystem::path(path))) {
    paths.emplace_back(path);
  } else {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(path)) {
      if (entry.is_regular_file()) paths.emplace_back(entry.path());
    }
  }
  return paths;
}

/// \brief Search file paths recursively.
/// This function calls the single path version of find_file_paths() for each
/// path in the given vector.
inline std::vector<std::string> find_file_paths(
    const std::vector<std::string>& paths) {
  std::vector<std::string> found_paths;
  for (const auto& p : paths) {
    const auto ret = find_file_paths(p);
    found_paths.insert(found_paths.end(), ret.begin(), ret.end());
  }
  return found_paths;
}

}  // namespace saltatlas::dndetail