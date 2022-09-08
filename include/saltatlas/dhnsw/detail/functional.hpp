// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

// TODO: Should have cereal includes elsewhere...
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

namespace saltatlas {

namespace detail {

// Want to make these more generic, so it doesn't have types returned by KNN
// hard-coded Requires multiple parameter packs though: one for data known at
// serialization time and one for data known at deserialization time

}  // namespace detail
}  // namespace saltatlas
