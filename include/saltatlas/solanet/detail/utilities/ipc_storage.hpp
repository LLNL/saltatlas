// Copyright 2020-2025 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <metall/detail/file.hpp>
#include <metall/detail/mmap.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "saltatlas/neo_dnnd/mpi.hpp"

namespace saltatlas::dndetail {

namespace {
namespace mdtl = metall::mtlldetail;
}

template <typename T>
struct ipc_mem_unique_ptr {
 public:
  ipc_mem_unique_ptr() = default;

  ipc_mem_unique_ptr(const std::string& name, int fd, T* addr, size_t length,
                     bool is_owner = false)
      : m_name(name),
        m_fd(fd),
        m_addr(addr),
        m_length(length),
        m_is_owner(is_owner) {}

  ~ipc_mem_unique_ptr() noexcept { priv_destroy(); }

  ipc_mem_unique_ptr(const ipc_mem_unique_ptr&)            = delete;
  ipc_mem_unique_ptr& operator=(const ipc_mem_unique_ptr&) = delete;

  ipc_mem_unique_ptr(ipc_mem_unique_ptr&& other) noexcept
      : m_name(std::move(other.m_name)),
        m_fd(other.m_fd),
        m_addr(other.m_addr),
        m_length(other.m_length),
        m_is_owner(other.m_is_owner) {
    other.m_fd       = -1;
    other.m_addr     = nullptr;
    other.m_length   = 0;
    other.m_is_owner = false;
  }

  ipc_mem_unique_ptr& operator=(ipc_mem_unique_ptr&& other) noexcept {
    if (this != &other) {
      priv_destroy();
      m_name     = std::move(other.m_name);
      m_fd       = other.m_fd;
      m_addr     = other.m_addr;
      m_length   = other.m_length;
      m_is_owner = other.m_is_owner;

      other.m_fd       = -1;
      other.m_addr     = nullptr;
      other.m_length   = 0;
      other.m_is_owner = false;
    }
    return *this;
  }

  std::string_view name() const { return m_name; }

  T* get() { return m_addr; }

  const T* get() const { return m_addr; }

  size_t size() const { return m_length; }

  bool is_owner() const { return m_is_owner; }

  void reset() { priv_destroy(); }

 private:
  void priv_destroy() {
    if (m_is_owner) {
      if (m_length > 0) {
        if (!mdtl::map_with_prot_none(m_addr, m_length * sizeof(T))) {
          std::cerr << "Failed to unmap shared memory region: " << m_name
                    << std::endl;
          std::abort();
        }
      }
    }

    if (m_addr != nullptr && m_length > 0) {
      mdtl::os_munmap(m_addr, m_length * sizeof(T));
    }

    if (m_fd != -1) {
      mdtl::os_close(m_fd);
    }

    if (m_is_owner) {
      shm_unlink(m_name.c_str());
    }

    m_addr   = nullptr;
    m_fd     = -1;
    m_length = 0;
    m_name.clear();
    m_is_owner = false;
  }

  std::string m_name{};
  int         m_fd{-1};
  T*          m_addr{nullptr};
  size_t      m_length{0};
  bool        m_is_owner{false};
};

template <typename T>
inline std::vector<ipc_mem_unique_ptr<T>> create_and_open_ipc_storage(
    const std::string& shm_name, const size_t length, mpi::communicator& comm) {
  const auto          node_size = comm.node_size();
  const auto          nlc_rank  = comm.node_local_rank();
  const auto          mem_size  = length * sizeof(T);
  std::vector<size_t> mem_sizes;
  comm.all_gather(mem_size, mem_sizes, false);

  std::vector<std::string> shm_names(node_size);
  for (int i = 0; i < node_size; ++i) {
    shm_names[i] = "/" + shm_name + "-" + std::to_string(i);
  }

  // Each rank creates its own shared memory region
  {
    int fd = shm_open(shm_names[nlc_rank].c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
      std::stringstream ss;
      ss << "shm_open (create) for " << shm_names[nlc_rank];
      perror(ss.str().c_str());
      comm.abort();
    }

    if (ftruncate(fd, mem_sizes[nlc_rank]) == -1) {
      perror("ftruncate");
      comm.abort();
    }

    close(fd);  // Will reopen all for mmap
  }
  comm.node_local_barrier();

  // Open and mmap all shared memory regions
  std::vector<ipc_mem_unique_ptr<T>> shm_regions(node_size);
  for (int i = 0; i < node_size; ++i) {
    // If the requested region size is zero, avoid mmap/munmap and create
    // an empty descriptor instead. Some ranks may have zero-sized
    // partitions; mapping size 0 can return implementation-defined results
    // and cause munmap/free issues later.
    if (mem_sizes[i] == 0) {
      // Close any fd if present (open below) and construct a null region.
      int flags = (i == nlc_rank) ? O_RDWR : O_RDONLY;
      int fd    = shm_open(shm_names[i].c_str(), flags, 0666);
      if (fd != -1) {
        close(fd);
      }
      shm_regions[i] =
          ipc_mem_unique_ptr<T>(shm_names[i], -1, nullptr, 0, i == nlc_rank);
      continue;
    }

    int flags = (i == nlc_rank) ? O_RDWR : O_RDONLY;
    int fd    = shm_open(shm_names[i].c_str(), flags, 0666);
    if (fd == -1) {
      std::stringstream ss;
      ss << "shm_open (access) for " << shm_names[i];
      perror(ss.str().c_str());
      comm.abort();
    }

    int prot = PROT_READ;
    if (i == nlc_rank) prot |= PROT_WRITE;

    auto* addr = mdtl::os_mmap(nullptr, mem_sizes[i], prot, MAP_SHARED, fd, 0);
    if (!addr) {
      comm.cerr0() << "Failed to mmap shared memory region: " << shm_names[i]
                   << std::endl;
      comm.abort();
    }
    shm_regions[i] =
        ipc_mem_unique_ptr<T>(shm_names[i], fd, static_cast<T*>(addr),
                              mem_sizes[i] / sizeof(T), i == nlc_rank);

#ifndef NDEBUG
    // Initialize the region with zero for debug
    if (i == nlc_rank) {
      std::memset(shm_regions[i].get(), 0, mem_sizes[i]);
    }
#endif
  }
  return shm_regions;
}

}  // namespace saltatlas::dndetail