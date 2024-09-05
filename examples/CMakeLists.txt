# Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
# saltatlas Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

#
# This function adds an example.
#
function(add_saltatlas_example example_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
endfunction()

add_saltatlas_example(binary_nn_benchmark)
add_saltatlas_example(space_wrapper_example)
add_saltatlas_example(dhnsw_example)
add_saltatlas_example(dhnsw_levenshtein)
add_saltatlas_example(dhnsw_ascii_levenshtein)
add_saltatlas_example(dhnsw_big_ann)
add_saltatlas_example(dhnsw_verbose_levenshtein)

if (SALTATLAS_USE_HDF5)
    add_saltatlas_example(benchmark_hdf5)
endif ()

#
# This function adds a DNND example with the given feature element type.
#
function(add_saltatlas_dnnd_example_feature_type example_name type_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}_${type_name}_features")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
    target_compile_definitions(${example_exe}
            PRIVATE "SALTATLAS_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE=${type_name}")
endfunction()

add_saltatlas_example(dnnd_simple)
add_saltatlas_example(dnnd_simple_custom_distance)
add_saltatlas_example(dnnd_simple_custom_point)
add_saltatlas_example(dnnd_levenshtein)

add_saltatlas_dnnd_example_feature_type(dnnd_bench float)
add_saltatlas_dnnd_example_feature_type(dnnd_bench uint8_t)
add_saltatlas_dnnd_example_feature_type(dnnd_bench double)

if (SALTATLAS_USE_METALL)
    add_saltatlas_example(dnnd_advanced)
endif ()

add_subdirectory(datasets)
