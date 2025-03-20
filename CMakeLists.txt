# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

# Works with 3.11 and tested through 3.15 (not tested yet)
cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------- #
# CMake policy
# ---------------------------------------------------------------------------- #
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.30")
    cmake_policy(SET CMP0167 NEW)
endif ()

# ---------------------------------------------------------------------------- #
# saltatlas configuration
# ---------------------------------------------------------------------------- #
project(
    saltatlas
    VERSION 0.1
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(SALTATLAS_MAIN_PROJECT ON)
endif ()

# Controls whether to set up install boilerplate for project and depencencies.
# Expects CMAKE_INSTALL_PREFIX to be set to a suitable directory.
option(JUST_INSTALL_SALTATLAS "Skip executables" OFF)
option(SALTATLAS_INSTALL "Generate the install target"
       ${SALTATLAS_MAIN_PROJECT}
)
option(SALTATLAS_USE_METALL "Use Metall" OFF)
option(SALTATLAS_DETERMINISM "Avoid shuffling for deterministic output" OFF)
option(SALTATLAS_USE_HDF5 "Include HDF5 and h5 read/write utilities" OFF)

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

# Check if the compiler is GCC
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.3.1")
        message(
            WARNING
                "Your GCC version may be too old. Please consider using GCC version 9.3.1 or higher."
        )
    endif ()
else ()
    message(
        WARNING
            "You are not using GCC. This project has only been tested with GCC. Please consider using GCC version 9.3.1 or higher."
    )
endif ()

include(FetchContent)

#
# OpenMP
#
find_package(OpenMP)

#
# Boost
#
if (NOT boost_POPULATED AND SALTATLAS_USE_METALL)
    cmake_policy(VERSION 3.30)
    cmake_policy(SET CMP0167 NEW)
    find_package(Boost 1.80)
    if (Boost_FOUND)
        message(STATUS ${PROJECT_NAME} " found boost include dirs: "
                       ${Boost_INCLUDE_DIRS}
        )
    else ()
        FetchContent_Declare(
            Boost
            URL https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.gz
        )
        FetchContent_MakeAvailable(Boost)
    endif ()
endif ()

#
# hnswlib
#
if (NOT hnswlib_POPULATED)
    find_package(hnswlib CONFIG QUIET)
    if (NOT hnswlib_FOUND)
        # Do not compile any hnswlib tests
        set(JUST_INSTALL_HNSWLIB ON)
        # Install hnswlib at ${CMAKE_INSTALL_PREFIX}
        set(HNSWLIB_INSTALL ${SALTATLAS_INSTALL})
        FetchContent_Declare(
            hnswlib
            GIT_REPOSITORY https://github.com/bwpriest/hnswlib.git
            GIT_TAG feature/cmake_facelift
        )
        FetchContent_MakeAvailable(hnswlib)
    endif ()
endif ()

#
# YGM
#
if (NOT ygm_POPULATED)
    find_package(ygm CONFIG QUIET)
    if (NOT ygm_FOUND)
        set(JUST_INSTALL_YGM TRUE)
        FetchContent_Declare(
            ygm
            GIT_REPOSITORY https://github.com/bwpriest/ygm.git
            GIT_TAG feature/cmake_3.30_tweaks
        )
        FetchContent_MakeAvailable(ygm)
    endif ()
endif ()

add_library(saltatlas INTERFACE)
add_library(saltatlas::saltatlas ALIAS saltatlas)
target_compile_features(saltatlas INTERFACE cxx_std_20)
target_include_directories(
    saltatlas INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    saltatlas INTERFACE ${SALTATLAS_CEREAL_TARGET} ygm::ygm hnswlib::hnswlib
)
if (SALTATLAS_DETERMINISM)
    message(STATUS ${PROJECT_NAME} " setting determinism")
    target_compile_definitions(saltatlas INTERFACE SALTATLAS_DETERMINISM)
endif ()

#
# saltatlas_h5_io (needs findable HDF5 >= v1.10.7)
#
if (SALTATLAS_USE_HDF5)
    find_package(saltatlas_h5_io CONFIG)
    if (NOT saltatlas_h5_io_FOUND)
        # Do not compile any saltatlas_h5_io tests
        set(JUST_INSTALL_SALTATLAS_H5_IO ON)
        # Install h5_cpp_io at ${CMAKE_INSTALL_PREFIX}
        set(SALTATLAS_H5_IO_INSTALL ${SALTATLAS_INSTALL})
        FetchContent_Declare(
            saltatlas_h5_io
            GIT_REPOSITORY https://github.com/LLNL/saltatlas_h5_io.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(saltatlas_h5_io)
    endif ()
    target_link_libraries(saltatlas INTERFACE saltatlas_h5_io::saltatlas_h5_io)
endif ()

#
# Metall
#
if (SALTATLAS_USE_METALL)
    find_package(Metall QUIET CONFIG)
    if (NOT Metall_FOUND)
        set(JUST_INSTALL_METALL_HEADER ON)
        FetchContent_Declare(
            Metall
            GIT_REPOSITORY https://github.com/LLNL/metall.git
            GIT_TAG master
        )
        FetchContent_MakeAvailable(Metall)
    endif ()

    find_package(Threads REQUIRED)
    target_include_directories(
        saltatlas INTERFACE $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
                            $<INSTALL_INTERFACE:include/boost>
    )
    target_link_libraries(
        saltatlas INTERFACE stdc++fs Threads::Threads Metall::Metall
    )
endif ()

if (SALTATLAS_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(SALTATLAS_EXPORT_TARGETS saltatlas)

    install(
        TARGETS ${SALTATLAS_EXPORT_TARGETS} ygm cereal
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/saltatlas
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # create version file
    write_basic_package_version_file(
        "${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY ExactVersion
    )

    # create config file
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
    )

endif ()

if (JUST_INSTALL_SALTATLAS)
    return()
endif ()

if (SALTATLAS_MAIN_PROJECT)
    add_subdirectory(examples)
    add_subdirectory(utility)
    add_subdirectory(test)
endif ()
