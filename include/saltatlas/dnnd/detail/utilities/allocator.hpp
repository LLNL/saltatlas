// Copyright 2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <scoped_allocator>

namespace saltatlas::dndetail {

template <typename Alloc, typename OtherT>
using other_allocator =
    typename std::allocator_traits<Alloc>::template rebind_alloc<OtherT>;

template <typename Alloc>
using scoped_allocator = typename std::scoped_allocator_adaptor<Alloc>;

template <typename Alloc, typename OtherT>
using other_scoped_allocator = scoped_allocator<other_allocator<Alloc, OtherT>>;

}  // namespace saltatlas::dndetail