// Copyright 2020-2026 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include "saltatlas/neo_dnnd/time_recorder.hpp"

namespace saltatlas {

static time_recorder& rec_time() {
  static time_recorder rec;
  return rec;
}

}  // namespace saltatlas