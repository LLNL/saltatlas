// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

inline void show_backtrace(int sig) {
  constexpr int MAX_FRAMES = 100;
  void*         array[MAX_FRAMES];
  const size_t  size = ::backtrace(array, MAX_FRAMES);
  ::fprintf(stderr, "Error: signal %d:\n", sig);
  ::backtrace_symbols_fd(array, size, STDERR_FILENO);
  ::exit(1);
}

inline void show_backtrace() {
  constexpr int MAX_FRAMES = 100;
  void*         array[MAX_FRAMES];
  const size_t  size = ::backtrace(array, MAX_FRAMES);
  if (size == 0) {
    return;
  }
  ::backtrace_symbols_fd(array, size, STDERR_FILENO);
  ::exit(1);
}