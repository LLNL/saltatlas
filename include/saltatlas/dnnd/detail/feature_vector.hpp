// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#if __has_include(<metall/container/vector.hpp>)
#ifndef SALTATLAS_DNND_USE_METALL_CONTAINER
#define SALTATLAS_DNND_USE_METALL_CONTAINER 1
#endif
#endif

#if SALTATLAS_DNND_USE_METALL_CONTAINER
#include <metall/container/vector.hpp>
#else
#include <vector>
#endif

#include <memory>
#include <saltatlas/dnnd/detail/utilities/allocator.hpp>

namespace saltatlas::dndetail {

namespace {
namespace container =
#if SALTATLAS_DNND_USE_METALL_CONTAINER
    metall::container;
#else
    std;
#endif
}  // namespace

template <typename Element, typename Allocator = std::allocator<Element>>
using feature_vector =
    container::vector<Element, other_allocator<Allocator, Element>>;

}  // namespace saltatlas::dndetail