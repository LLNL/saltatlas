# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.29)

# ---------------------------------------------------------------------------- #
# CMake policy
# ---------------------------------------------------------------------------- #
# When using the URL download method with the ExternalProject_Add() or
# FetchContent_Declare() commands, sets the timestamps of extracted contents
# to the time of extraction.
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif ()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif ()

# ---------------------------------------------------------------------------- #
# saltatlas configuration
# ---------------------------------------------------------------------------- #
set(lang_list "CXX")
if(SALTATLAS_USE_CUVS)
  list(APPEND lang_list "CUDA")
elseif(SALTATLAS_USE_APU)
  list(APPEND lang_list "HIP")
endif()

project(
        saltatlas
        VERSION 0.1
        LANGUAGES ${lang_list}
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(SALTATLAS_MAIN_PROJECT ON)
endif ()

# Controls whether to set up install boilerplate for project and depencencies.
# Expects CMAKE_INSTALL_PREFIX to be set to a suitable directory.
option(JUST_INSTALL_SALTATLAS "Skip executables" OFF)
option(SALTATLAS_USE_HNSWLIB "Use Hnswlib" ON)
option(SALTATLAS_USE_METALL "Use Metall" ON)
option(SALTATLAS_DETERMINISM "Avoid shuffling for deterministic output" OFF)
# User options to specify Boost source or URL
set(BOOST_SOURCE_DIR CACHE STRING "Path to Boost libraries directory already uncompressed. Must be a version that supports CMake.")
set(BOOST_FETCH_URL CACHE STRING "URL or file path to an archived Boost source. Must be a version that supports CMake.")
set(HNSWLIB_FETCH_GIT
        "https://github.com/bwpriest/hnswlib.git"
        CACHE STRING "GIT to fetch hnswlib")
set(HNSWLIB_GIT_BRANCH
        "feature/cmake_facelift"
        CACHE STRING "Name of GIT branch for hnswlib")
set(YGM_FETCH_GIT
        "https://github.com/llnl/ygm.git"
        CACHE STRING "GIT to fetch ygm")
set(YGM_GIT_BRANCH
        "master"
        CACHE STRING "Name of GIT branch for ygm")
set(METALL_FETCH_GIT
        "https://github.com/llnl/metall.git"
        CACHE STRING "GIT to fetch Metall")
set(METALL_GIT_BRANCH
        "develop"
        CACHE STRING "Name of GIT branch for Metall")
set(CEREAL_FETCH_GIT
    "https://github.com/USCiLab/cereal.git"
    CACHE STRING "GIT to fetch Cereal")
# Link Boost UUID with libatomic fails on LC machines
option(BOOST_UUID_LINK_LIBATOMIC "Link Boost UUID with libatomic" OFF)
option(SALTATLAS_BUILD_TEST "Build the test" OFF)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if (SALTATLAS_MAIN_PROJECT)

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app Note this needs to be done
    # in the main CMakeLists since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)
endif ()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
    message(
            FATAL_ERROR
            "You cannot build in a source directory "
            "(or any directory with a CMakeLists.txt file). "
            "Please make a build subdirectory. "
            "Feel free to remove CMakeCache.txt and CMakeFiles."
    )
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE is set as Release")
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Thread
find_package(Threads REQUIRED)

#
# OpenMP
#
find_package(OpenMP)
function(setup_omp_target target_name)
    if (NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP CXX not found!")
    else ()
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endfunction()

#
# MPI
#
find_package(MPI REQUIRED)
function(setup_mpi_target target_name)
    target_include_directories(${target_name} PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
endfunction()

# CUDA
option(SALTATLAS_USE_CUVS "Use CUDA" OFF)
if (SALTATLAS_USE_CUVS)
    find_package(CUDAToolkit REQUIRED)
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90) # Matrix is 90
    endif ()
    enable_language(CUDA)

    # find_package(rmm REQUIRED)
    find_package(raft REQUIRED)
    find_package(cuvs REQUIRED)


    if (cuvs_VERSION VERSION_EQUAL "25.02.01")
        # Need to be enabled to cuvs v25.02.01
        add_library(rmm_logger OBJECT)
        target_link_libraries(rmm_logger PRIVATE rmm::rmm_logger_impl)
    endif ()
endif ()

function(setup_cuvs_target target_name)
    target_include_directories(
            ${target_name} PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(
            ${target_name}
            PUBLIC Threads::Threads CUDA::cudart ${NCCL_LIBRARY} cuvs::cuvs
            raft::raft rmm::rmm
            # rmm_logger # need for cuvs v25.02.01
    )
endfunction()

#
# HIP
#
option(SALTATLAS_USE_APU "Use HIP" OFF)
if (SALTATLAS_USE_APU)
    find_package(hip REQUIRED)

    file(GLOB CUDA_SRC_FILES "src/*.cu")
    set_source_files_properties(${CUDA_SRC_FILES} PROPERTIES LANGUAGE HIP)

    if (NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx942") #  'gfx942' for Tuolumne
    endif ()

    find_package(cuvs REQUIRED)
    find_package(raft REQUIRED)

    # # ------------- configure rapids-cmake --------------#
    # include(./cmake/thirdparty/fetch_rapids.cmake)
    # include(rapids-cmake)
    # include(rapids-cpm)
    # include(rapids-export)
    # include(rapids-find)
    # include(rapids-hip)
    # rapids_hip_init_architectures(hipvs_cxx_examples)

    # # ------------- configure cuvs -----------------#
    # rapids_cpm_init()
    # set(BUILD_CUVS_C_LIBRARY OFF)
    # include(./cmake/thirdparty/get_cuvs.cmake)

endif ()

function(setup_hip_target target_name)
    target_link_libraries(
            ${target_name} PUBLIC hip::device
    )
endfunction()

function(setup_hipvs_target target_name)
    target_link_libraries(
            ${target_name} PUBLIC cuvs::cuvs rmm::rmm_logger_impl

    )
endfunction()

#
# Boost
#
include(fetch_boost)

# List of boost components this project depends on
set(BOOST_COMPS json unordered container static_string)
if (SALTATLAS_USE_METALL)
    # Metall requires these
    # Extend BOOST_COMPS
    list(APPEND BOOST_COMPS interprocess property_tree uuid)
endif ()

# We do not check if Boost is already populated, since fetch_boost() does that.
if (BOOST_SOURCE_DIR)
    fetch_boost(SOURCE ${BOOST_SOURCE_DIR} COMPONENTS ${BOOST_COMPS})
elseif (BOOST_FETCH_URL)
    fetch_boost(URL ${BOOST_FETCH_URL} COMPONENTS ${BOOST_COMPS})
else ()
    # Use default Boost version if not specified by the user
    fetch_boost(URL "https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.gz"
            COMPONENTS ${BOOST_COMPS})
endif ()
# Manually set boost_POPULATED to TRUE so that subsequent calls to fetch_boost()
# know that Boost is already available.
set(boost_POPULATED TRUE)

function(setup_boost_target target_name)
    target_link_libraries(${target_name} PUBLIC ${BOOST_LIBS})
endfunction()

#
# Cereal
#
# if (NOT cereal_POPULATED)
#     find_package(cereal CONFIG QUIET)
#     if (NOT cereal_FOUND)
#         set(JUST_INSTALL_CEREAL TRUE)
#         FetchContent_Declare(
#             cereal
#             GIT_REPOSITORY ${CEREAL_FETCH_GIT}
#             GIT_TAG v1.3.2
#         )
#         FetchContent_MakeAvailable(cereal)
#     endif ()
# endif ()

#
# hnswlib
#
if (SALTATLAS_USE_HNSWLIB)
    # Do not compile any hnswlib tests
    set(JUST_INSTALL_HNSWLIB ON)
    FetchContent_Declare(
            hnswlib
            GIT_REPOSITORY ${HNSWLIB_FETCH_GIT}
            GIT_TAG ${HNSWLIB_GIT_BRANCH}
    )
    FetchContent_MakeAvailable(hnswlib)
endif ()

#
# spdlog
#
find_package(spdlog CONFIG QUIET)
if (TARGET spdlog::spdlog OR TARGET spdlog::spdlog_header_only)
    set(spdlog_FOUND TRUE)
endif ()
if (NOT spdlog_FOUND)
    set(SPDLOG_INSTALL On)
    FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(spdlog)
endif ()

# Compatibility targets for dependencies that link plain `spdlog` or
# `spdlog_header_only` instead of namespaced imported targets.
if (TARGET spdlog::spdlog_header_only AND NOT TARGET spdlog_header_only)
    add_library(spdlog_header_only INTERFACE)
    target_link_libraries(spdlog_header_only INTERFACE spdlog::spdlog_header_only)
endif ()
if (TARGET spdlog::spdlog AND NOT TARGET spdlog)
    add_library(spdlog INTERFACE)
    target_link_libraries(spdlog INTERFACE spdlog::spdlog)
endif ()

function(setup_spdlog_target target_name)
    if (TARGET spdlog::spdlog_header_only)
        target_link_libraries(${target_name} PRIVATE spdlog::spdlog_header_only)
    elseif (TARGET spdlog_header_only)
        target_link_libraries(${target_name} PRIVATE spdlog_header_only)
    elseif (TARGET spdlog::spdlog)
        target_link_libraries(${target_name} PRIVATE spdlog::spdlog)
    elseif (TARGET spdlog)
        target_link_libraries(${target_name} PRIVATE spdlog)
    else ()
        message(FATAL_ERROR "No usable spdlog target found for ${target_name}")
    endif ()
endfunction()

#
# YGM
#
# Set this flag manually so that YGM does not try to fetch spdlog again.
set(spdlog_POPULATED TRUE)
FetchContent_Declare(
        ygm
        GIT_REPOSITORY ${YGM_FETCH_GIT}
        GIT_TAG ${YGM_GIT_BRANCH}
        SYSTEM
)
FetchContent_MakeAvailable(ygm)

#
# SALTATLAS INTERFACE
#
add_library(saltatlas INTERFACE)
add_library(saltatlas::saltatlas ALIAS saltatlas)
target_compile_features(saltatlas INTERFACE cxx_std_20)
target_include_directories(
        saltatlas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
        saltatlas INTERFACE ${SALTATLAS_CEREAL_TARGET} ygm::ygm ${BOOST_LIBS}
)
if (SALTATLAS_USE_HNSWLIB)
    target_link_libraries(saltatlas INTERFACE hnswlib::hnswlib)
endif ()
if (SALTATLAS_DETERMINISM)
    message(STATUS ${PROJECT_NAME} " setting determinism")
    target_compile_definitions(saltatlas INTERFACE SALTATLAS_DETERMINISM)
endif ()

#
# Metall
#
if (SALTATLAS_USE_METALL)
    set(JUST_INSTALL_METALL_HEADER ON)
    FetchContent_Declare(
            Metall
            GIT_REPOSITORY ${METALL_FETCH_GIT}
            GIT_TAG ${METALL_GIT_BRANCH}
    )
    FetchContent_MakeAvailable(Metall)
    find_package(Threads REQUIRED)

    target_link_libraries(saltatlas INTERFACE Metall::Metall)
    target_link_libraries(saltatlas INTERFACE Threads::Threads)
endif ()
function(setup_metall_target target_name)
    target_link_libraries(
            ${target_name} PRIVATE Threads::Threads Metall::Metall
    )
    setup_boost_target(${target_name})
endfunction()


function(setup_metall_target target_name)
    if (SALTATLAS_USE_METALL)
        target_link_libraries(${target_name} PRIVATE Metall::Metall)
        target_link_libraries(${target_name} PRIVATE Threads::Threads)
    endif ()
endfunction()

if (JUST_INSTALL_SALTATLAS)
    return()
endif ()

# Enable exports only in Debug build
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_ENABLE_EXPORTS TRUE)
endif ()


if (SALTATLAS_MAIN_PROJECT)
    function(setup_common_compile_options target_name)
        # Common
        target_compile_options(${target_name} PRIVATE -Wall)
        # if (SALTATLAS_USE_CUVS)
        #     target_compile_definitions(${target_name} PUBLIC SALTATLAS_USE_CUVS)
        # endif ()

        # Debug
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-O0>)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-g3>)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-Wextra>)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-pg>)
        endif ()

        # Release
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-O3>)
        #target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-Ofast>)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-DNDEBUG>)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release>:-march=native>)

        # Release with debug info
        target_compile_options(
                ${target_name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3>
        )
        target_compile_options(
                ${target_name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-g3>
        )
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_options(
                    ${target_name} PRIVATE $<$<CONFIG:RelWithDebInfo>:-pg>
            )
        endif ()
    endfunction()

    function(add_general_exe target_name src_name)
        add_executable(${target_name} "${src_name}")
        target_include_directories(
                ${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/include
        )
        setup_common_compile_options(${target_name})

        setup_omp_target(${target_name})
        setup_boost_target(${target_name})
        if (SALTATLAS_USE_METALL)
            setup_metall_target(${target_name})
        endif ()
        setup_spdlog_target(${target_name})

        # Link to rt if on Linux
        if (UNIX AND NOT APPLE)
            target_link_libraries(${target_name} PRIVATE rt)
        endif ()
    endfunction()

    function(add_ygm_exe target_name)
        add_general_exe(${target_name} "${target_name}.cpp")
        setup_mpi_target(${target_name})
        target_link_libraries(
                ${target_name} PRIVATE ${SALTATLAS_CEREAL_TARGET} ygm::ygm ${BOOST_LIBS})
    endfunction()

    #
    # This function adds an executable with the given feature element type.
    #
    function(add_fe_typed_exe target_name type_name)
        set(typed_target_name "${target_name}_${type_name}_features")
        add_general_exe(${typed_target_name} "${target_name}.cpp")
        target_compile_definitions(
                ${typed_target_name}
                PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${type_name}"
        )
        set(target_name
                ${typed_target_name}
                PARENT_SCOPE
        )
    endfunction()

    #
    # This function adds an MPI target with the given feature element type.
    #
    function(add_fe_typed_mpi_exe target_name type_name)
        set(typed_target_name "${target_name}_${type_name}_features")
        add_general_exe(${typed_target_name} "${target_name}.cpp")
        setup_mpi_target(${typed_target_name})
        target_compile_definitions(
                ${typed_target_name}
                PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${type_name}"
        )
        set(target_name
                ${typed_target_name}
                PARENT_SCOPE
        )
    endfunction()

    function(add_fe_typed_ygm_exe target_name type_name)
        set(typed_target_name "${target_name}_${type_name}_features")
        add_general_exe(${typed_target_name} "${target_name}.cpp")
        setup_mpi_target(${typed_target_name})
        target_link_libraries(
                ${typed_target_name} PRIVATE ${SALTATLAS_CEREAL_TARGET} ygm::ygm ${BOOST_LIBS})
        target_compile_definitions(
                ${typed_target_name}
                PRIVATE "SALTATLAS_FEATURE_ELEMENT_TYPE=${type_name}"
        )
        set(target_name
                ${typed_target_name}
                PARENT_SCOPE
        )
    endfunction()

    # add_subdirectory(src)
    add_subdirectory(examples)
    add_subdirectory(tools)
    # if (SALTATLAS_BUILD_TEST)
    #     enable_testing()
    #     add_subdirectory(test)
    # endif ()
endif ()
