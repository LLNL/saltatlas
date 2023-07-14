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
# This function adds an example with the given feature type.
#
function(add_saltatlas_example_with_feature_type example_name type_name)
    set(example_source "${example_name}.cpp")
    set(example_exe "${example_name}_${type_name}_features")
    add_executable(${example_exe} ${example_source})
    target_link_libraries(${example_exe} PUBLIC saltatlas)
    target_compile_definitions(${example_exe}
            PRIVATE "SALTATLAS_DNND_EXAMPLE_FEATURE_ELEMENT_TYPE=${type_name}")
endfunction()

if (SALTATLAS_USE_METALL)
    add_saltatlas_example(dnnd_example)
    add_saltatlas_example(dnnd_string_example)
    add_saltatlas_example(dnnd_pm_const_example)
    add_saltatlas_example(dnnd_pm_optimize_example)
    add_saltatlas_example(dnnd_pm_query_example)

    add_saltatlas_example_with_feature_type(dnnd_pm_const_example float)
    add_saltatlas_example_with_feature_type(dnnd_pm_optimize_example float)
    add_saltatlas_example_with_feature_type(dnnd_pm_query_example float)

    add_saltatlas_example_with_feature_type(dnnd_pm_const_example uint8_t)
    add_saltatlas_example_with_feature_type(dnnd_pm_optimize_example uint8_t)
    add_saltatlas_example_with_feature_type(dnnd_pm_query_example uint8_t)

    add_saltatlas_example_with_feature_type(dnnd_pm_const_example double)
    add_saltatlas_example_with_feature_type(dnnd_pm_optimize_example double)
    add_saltatlas_example_with_feature_type(dnnd_pm_query_example double)
endif ()

add_subdirectory(datasets)
