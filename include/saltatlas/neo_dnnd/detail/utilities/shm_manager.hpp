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
#include <string>
#include <string_view>
#include <vector>

#include <metall/detail/file.hpp>
#include <metall/detail/mmap.hpp>

#include "../../mpi.hpp"

namespace saltatlas::dndetail {

namespace {
namespace mdtl = metall::mtlldetail;
}

template <typename T>
struct shm_unique_ptr {
 public:
  shm_unique_ptr() = default;

  shm_unique_ptr(const std::string& name, int fd, T* addr, size_t length,
                 bool is_owner = false)
      : m_name(name),
        m_fd(fd),
        m_addr(addr),
        m_length(length),
        m_is_owner(is_owner) {}

  ~shm_unique_ptr() noexcept { priv_destroy(); }

  shm_unique_ptr(const shm_unique_ptr&)            = delete;
  shm_unique_ptr& operator=(const shm_unique_ptr&) = delete;

  shm_unique_ptr(shm_unique_ptr&& other) noexcept
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

  shm_unique_ptr& operator=(shm_unique_ptr&& other) noexcept {
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
      if (!mdtl::map_with_prot_none(m_addr, m_length * sizeof(T))) {
        std::cerr << "Failed to unmap shared memory region: " << m_name
                  << std::endl;
        std::abort();
      }
    }

    if (m_addr != nullptr) {
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
inline std::vector<shm_unique_ptr<T>> create_and_open_shm(
    const std::string& shm_name, const size_t length, mpi::communicator& comm) {
  const auto          lc_size  = comm.local_size();
  const auto          lc_rank  = comm.local_rank();
  const auto          mem_size = length * sizeof(T);
  std::vector<size_t> mem_sizes;
  comm.all_gather(mem_size, mem_sizes, false);

  std::vector<std::string> shm_names(lc_size);
  for (int i = 0; i < lc_size; ++i) {
    shm_names[i] = "/" + shm_name + "-" + std::to_string(i);
  }

  // Each rank creates its own shared memory region
  {
    int fd = shm_open(shm_names[lc_rank].c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
      perror("shm_open (create)");
      comm.abort();
    }

    if (ftruncate(fd, mem_sizes[lc_rank]) == -1) {
      perror("ftruncate");
      comm.abort();
    }

    close(fd);  // Will reopen all for mmap
  }
  comm.local_barrier();

  // Open and mmap all shared memory regions
  std::vector<shm_unique_ptr<T>> shm_regions(lc_size);
  for (int i = 0; i < lc_size; ++i) {
    int flags = (i == lc_rank) ? O_RDWR : O_RDONLY;
    int fd    = shm_open(shm_names[i].c_str(), flags, 0666);
    if (fd == -1) {
      perror("shm_open (access)");
      comm.abort();
    }

    int prot = PROT_READ;
    if (i == lc_rank) prot |= PROT_WRITE;

    auto* addr = mdtl::os_mmap(nullptr, mem_sizes[i], prot, MAP_SHARED, fd, 0);
    if (!addr) {
      comm.cerr0() << "Failed to mmap shared memory region: " << shm_names[i]
                   << std::endl;
      comm.abort();
    }
    shm_regions[i] = shm_unique_ptr<T>(shm_names[i], fd, static_cast<T*>(addr),
                                       mem_sizes[i] / sizeof(T), i == lc_rank);
  }
  return shm_regions;
}

}  // namespace saltatlas::dnnd3::shm_manager