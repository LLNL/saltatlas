// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "../mpi.hpp"
#include "saltatlas/common/detail/utilities/string_cast.hpp"

namespace saltatlas::dndetail {

namespace {
namespace mpi = saltatlas::mpi;
}

template <typename _id_type, typename _feature_elem_type>
class point_reader {
 public:
  using id_type           = _id_type;
  using feature_elem_type = _feature_elem_type;

  point_reader()  = default;
  ~point_reader() = default;

  /// When reading files w/o ID, file paths are sorted by lexicographical order.
  /// IDs are assigned after the sorting.
  template <typename partitioner_type>
  static std::pair<std::vector<id_type>,
                   std::vector<std::vector<feature_elem_type>>>
  read(const std::vector<std::filesystem::path>& paths,
       const std::string_view format, const partitioner_type& partitioner,
       mpi::communicator& comm) {
    if (format == "wsv") {
      return priv_read_wsv(priv_get_file_list(paths), partitioner, comm);
    } else if (format == "wsv-id") {
      return priv_read_wsv_with_id(priv_get_file_list(paths), partitioner,
                                   comm);
    } else if (format == "bin") {
      return priv_read_bin(priv_get_file_list(paths), partitioner, comm);
    } else if (format == "bin-id") {
      return priv_read_bin_with_id(priv_get_file_list(paths), partitioner,
                                   comm);
    } else {
      comm.cerr0() << "Unknown dataset format: " << format << std::endl;
      comm.abort();
    }
    return {};
  }

 private:
  static int priv_worker(const std::size_t i, const int num_ranks) {
    return hash<>{}(i) % num_ranks;
  }

  static std::vector<std::filesystem::path> priv_get_file_list(
      const std::vector<std::filesystem::path>& paths) {
    std::vector<std::filesystem::path> file_list;
    for (const auto& path : paths) {
      const auto fl = priv_get_file_list(path);
      file_list.insert(file_list.end(), fl.begin(), fl.end());
    }
    return file_list;
  }

  static std::vector<std::filesystem::path> priv_get_file_list(
      const std::filesystem::path& path) {
    std::vector<std::filesystem::path> file_list;
    // If 'path' is a file, return it.
    // If 'path' is a directory, return all files in it.
    if (std::filesystem::is_regular_file(path)) {
      file_list.push_back(path);
    } else if (std::filesystem::is_directory(path)) {
      for (const auto& entry :
           std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
          file_list.push_back(entry.path());
        }
      }
    }
    return file_list;
  }

  template <typename partitioner_type>
  static auto priv_read_bin(std::vector<std::filesystem::path> paths,
                            const partitioner_type&            partitioner,
                            mpi::communicator&                 comm) {
    std::sort(paths.begin(), paths.end());
    for (int i = 0; i < paths.size(); ++i) {
      comm.cout0() << i << " " << paths[i] << std::endl;
    }

    std::vector<std::size_t> offsets(paths.size(), 0);
    if (comm.rank() == 0) {
      for (int fi = 0; fi < paths.size(); ++fi) {
        std::ifstream ifs(paths[fi], std::ios::binary);
        if (!ifs) {
          comm.cerr0() << "Failed to open " << paths[fi] << std::endl;
          comm.abort();
        }

        uint64_t num_points;
        ifs.read(reinterpret_cast<char*>(&num_points), sizeof(uint64_t));
        offsets.at(fi) = num_points;
      }
      std::partial_sum(offsets.begin(), offsets.end() - 1, offsets.begin() + 1);
    }
    for (std::size_t i = 0; i < paths.size(); ++i) {
      comm.bcast(offsets[i], 0);
    }

    std::vector<std::vector<id_type>>           read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());

    uint64_t num_points = 0;
    uint64_t num_dims   = 0;
    for (int fi = 0; fi < paths.size(); ++fi) {
      if (priv_worker(fi, comm.size()) != comm.rank()) {
        continue;
      }

      std::ifstream ifs(paths[fi], std::ios::binary);
      if (!ifs) {
        comm.cerr0() << "Failed to open " << paths[fi] << std::endl;
        comm.abort();
      }

      uint64_t np;
      uint64_t nd;
      ifs.read(reinterpret_cast<char*>(&np), sizeof(uint64_t));
      ifs.read(reinterpret_cast<char*>(&nd), sizeof(uint64_t));
      if (num_points == 0) {
        num_dims = nd;
      } else if (nd != num_dims) {
        comm.cerr() << "All files must have the same #of dimensions"
                    << std::endl;
        comm.abort();
      }
      num_points += np;

      for (id_type pid = 0; pid < np; ++pid) {
        std::vector<feature_elem_type> feature(nd);
        ifs.read(reinterpret_cast<char*>(feature.data()),
                 nd * sizeof(feature_elem_type));
        const auto gpid  = pid + offsets[fi];
        const auto owner = partitioner(gpid);
        read_ids.at(owner).push_back(gpid);
        read_features.at(owner).insert(read_features.at(owner).end(),
                                       feature.begin(), feature.end());
      }
    }
    if (comm.all_reduce_min(num_dims) != num_dims) {
      comm.cerr() << "All files must have the same #of dimensions" << std::endl;
      comm.abort();
    }
    comm.cout0() << "#of dims: " << num_dims << std::endl;
    comm.cout0() << "#of points: " << comm.all_reduce_sum(num_points)
                 << std::endl;

    return priv_distribute_points(std::move(read_ids), std::move(read_features),
                                  num_dims, comm);
  }

  template <typename partitioner_type>
  static auto priv_read_bin_with_id(std::vector<std::filesystem::path> paths,
                                    const partitioner_type& partitioner,
                                    mpi::communicator&      comm) {
    std::vector<std::vector<id_type>>           read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());

    uint64_t num_points = 0;
    uint64_t num_dims   = 0;
    for (int fi = 0; fi < paths.size(); ++fi) {
      if (priv_worker(fi, comm.size()) != comm.rank()) {
        continue;
      }

      std::ifstream ifs(paths[fi], std::ios::binary);
      if (!ifs) {
        comm.cerr0() << "Failed to open " << paths[fi] << std::endl;
        comm.abort();
      }

      uint64_t np;
      uint64_t nd;
      ifs.read(reinterpret_cast<char*>(&np), sizeof(uint64_t));
      ifs.read(reinterpret_cast<char*>(&nd), sizeof(uint64_t));
      if (num_points == 0) {
        num_dims = nd;
      } else if (nd != num_dims) {
        comm.cerr() << "All files must have the same #of dimensions"
                    << std::endl;
        comm.abort();
      }
      num_points += np;

      for (size_t pi = 0; pi < np; ++pi) {
        id_type pid = 0;
        ifs.read(reinterpret_cast<char*>(pid), sizeof(id_type));
        std::vector<feature_elem_type> feature(nd);
        ifs.read(reinterpret_cast<char*>(feature.data()),
                 nd * sizeof(feature_elem_type));
        const auto owner = partitioner(pid);
        read_ids.at(owner).push_back(pid);
        read_features.at(owner).insert(read_features.at(owner).end(),
                                       feature.begin(), feature.end());
      }
    }
    if (comm.all_reduce_min(num_dims) != num_dims) {
      comm.cerr() << "All files must have the same #of dimensions" << std::endl;
      comm.abort();
    }
    comm.cout0() << "#of dims: " << num_dims << std::endl;
    comm.cout0() << "#of points: " << comm.all_reduce_sum(num_points)
                 << std::endl;

    return priv_distribute_points(std::move(read_ids), std::move(read_features),
                                  num_dims, comm);
  }

  template <typename partitioner_type>
  static auto priv_read_wsv(std::vector<std::filesystem::path> paths,
                            const partitioner_type&            partitioner,
                            mpi::communicator&                 comm) {
    std::sort(paths.begin(), paths.end());
    for (int i = 0; i < paths.size(); ++i) {
      comm.cout0() << i << " " << paths[i] << std::endl;
    }

    const std::vector<std::size_t> offsets =
        priv_convert_to_offsets(priv_count_lines_in_files(paths, comm));

    comm.cout0() << "Reading points w/o ID..." << std::endl;
    std::vector<std::vector<id_type>>           read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());
    std::size_t                                 dims = 0;
    for (int i = 0; i < int(paths.size()); ++i) {
      if (priv_worker(i, comm.size()) != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        comm.cerr() << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      id_type     pid = offsets[i];
      std::string line;
      while (std::getline(ifs, line)) {
        auto feature = detail::str_split<feature_elem_type>(line);
        if (dims == 0) {
          dims = feature.size();
        } else {
          if (dims != feature.size()) {
            comm.cerr() << "Inconsistent number of dimensions: " << dims
                        << " vs " << feature.size() << std::endl;
            comm.abort();
          }
        }
        read_ids[partitioner(pid)].push_back(pid);
        read_features[partitioner(pid)].insert(
            read_features[partitioner(pid)].end(), feature.begin(),
            feature.end());
        ++pid;
      }
    }
    comm.barrier();

    comm.bcast(dims, priv_worker(0, comm.size()));

    return priv_distribute_points(std::move(read_ids), std::move(read_features),
                                  dims, comm);
  }

  template <typename partitioner_type>
  static auto priv_read_wsv_with_id(
      const std::vector<std::filesystem::path>& paths,
      const partitioner_type& partitioner, mpi::communicator& comm) {
    std::vector<std::vector<id_type>>           read_ids(comm.size());
    std::vector<std::vector<feature_elem_type>> read_features(comm.size());
    std::size_t                                 dims = 0;
    std::unordered_set<id_type>                 id_set;

    comm.cout0() << "Reading points with ID..." << std::endl;
    for (int i = 0; i < int(paths.size()); ++i) {
      if (priv_worker(i, comm.size()) != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        comm.cerr() << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      std::string line;
      while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        id_type           pid;
        ss >> pid;
        if (id_set.count(pid) > 0) {
          comm.cerr() << "Duplicate ID: " << pid << std::endl;
          comm.abort();
        } else {
          id_set.insert(pid);
        }
        line         = ss.str().substr(ss.tellg());  // Remove the ID part.
        auto feature = detail::str_split<feature_elem_type>(line);
        if (dims == 0) {
          dims = feature.size();
        } else {
          if (dims != feature.size()) {
            comm.cerr() << "Inconsistent number of dimensions: " << dims
                        << " vs " << feature.size() << std::endl;
            comm.abort();
          }
        }
        const auto owner = partitioner(pid);
        read_ids[owner].push_back(pid);
        read_features[owner].insert(read_features[owner].end(), feature.begin(),
                                    feature.end());
      }
    }
    id_set.clear();
    id_set.rehash(0);
    comm.barrier();

    comm.bcast(dims, 0);

    return priv_distribute_points(std::move(read_ids), std::move(read_features),
                                  dims, comm);
  }

  static std::vector<std::size_t> priv_count_lines_in_files(
      const std::vector<std::filesystem::path>& paths,
      mpi::communicator&                        comm) {
    comm.cout0() << "Counting data..." << std::endl;

    std::vector<std::size_t> num_lines(paths.size());
    for (int i = 0; i < int(paths.size()); ++i) {
      if (priv_worker(i, comm.size()) != comm.rank()) {
        continue;
      }
      std::ifstream ifs(paths[i]);
      if (ifs) {
        // std::cout << "Opened " << paths[i] << std::endl;
      } else {
        comm.cerr() << "Failed to open " << paths[i] << std::endl;
        comm.abort();
      }

      std::string line;
      while (std::getline(ifs, line)) {
        ++num_lines[i];
      }
    }
    comm.barrier();

    // broadcast the number of lines in each file.
    for (std::size_t i = 0; i < paths.size(); ++i) {
      comm.bcast(num_lines[i], priv_worker(i, comm.size()));
    }

    const auto total_num_lines =
        std::accumulate(num_lines.cbegin(), num_lines.cend(), 0);

    comm.cout0() << "#of total lines: " << total_num_lines << std::endl;
    if (std::numeric_limits<id_type>::max() < total_num_lines) {
      comm.cerr0() << "Too small ID type: " << typeid(id_type).name()
                   << std::endl;
      comm.abort();
    }

    return num_lines;
  }

  static std::vector<std::size_t> priv_convert_to_offsets(
      const std::vector<std::size_t>& num_lines) {
    std::vector<std::size_t> offsets(num_lines.size());
    std::partial_sum(num_lines.cbegin(), num_lines.cend(), offsets.begin());
    offsets.insert(offsets.begin(), 0);
    return offsets;
  }
  static std::pair<std::vector<id_type>,
                   std::vector<std::vector<feature_elem_type>>>
  priv_distribute_points(
      std::vector<std::vector<id_type>>&&           read_ids,
      std::vector<std::vector<feature_elem_type>>&& read_features,
      const std::size_t dims, mpi::communicator& comm) {
    assert(read_ids.size() == std::size_t(comm.size()));
    assert(read_features.size() == std::size_t(comm.size()));

    comm.cout0() << "Distributing read points..." << std::endl;

    std::size_t num_assigned_points = 0;
    for (int r = 0; r < comm.size(); ++r) {
      std::size_t n = read_ids[r].size();
      DNND2_CHECK_MPI(::MPI_Reduce(
          &n, (r == comm.rank()) ? &num_assigned_points : nullptr, 1,
          mpi::data_type::get<std::size_t>(), MPI_SUM, r, comm.comm()));
    }

    std::vector<id_type>                        assigned_ids;
    std::vector<std::vector<feature_elem_type>> assigned_fvs;
    assigned_ids.reserve(num_assigned_points);
    assigned_fvs.reserve(num_assigned_points);

    mpi::pair_wise_all_to_all(
        comm.size(), comm.rank(),
        [&](const int pair_rank) {
          std::vector<id_type> ids_recv_buf;
          comm.sendrecv_arb_size(pair_rank, std::move(read_ids[pair_rank]),
                                 ids_recv_buf);

          std::vector<feature_elem_type> features_recv_buf;
          comm.sendrecv_arb_size(pair_rank, std::move(read_features[pair_rank]),
                                 features_recv_buf);

          const auto num_recv_points = ids_recv_buf.size();
          for (std::size_t i = 0; i < num_recv_points; ++i) {
            assigned_ids.push_back(ids_recv_buf[i]);
            auto& f = assigned_fvs.emplace_back(dims);
            assert(f.size() == dims);
            std::memcpy(f.data(), features_recv_buf.data() + i * dims,
                        dims * sizeof(feature_elem_type));
          }
        },
        comm.comm());
    comm.barrier();
    return std::make_pair(std::move(assigned_ids), std::move(assigned_fvs));
  }
};
}  // namespace saltatlas::dndetail