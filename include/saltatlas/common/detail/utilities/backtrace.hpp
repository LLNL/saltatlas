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
  void *array[32];
  const size_t size = ::backtrace(array, 32);
  ::fprintf(stderr, "Error: signal %d:\n", sig);
  ::backtrace_symbols_fd(array, size, STDERR_FILENO);
  ::exit(1);
}
