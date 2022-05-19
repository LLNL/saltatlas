// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

namespace saltatlas::dndetail {

template <typename Element, typename Allocator>
using feature_vector =
    std::vector<Element, other_allocator<Allocator, Element>>;

}  // namespace saltatlas::dndetail