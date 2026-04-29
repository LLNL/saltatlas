// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <metall/utility/mutex.hpp>

namespace saltatlas::dndetail {
namespace mutex {
static constexpr int k_num_mutexes = 1024;
using metall::utility::mutex::mutex_lock;
}  // namespace mutex
}  // namespace saltatlas::dndetail