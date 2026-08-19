if (SALTATLAS_USE_APU)
    add_fe_typed_exe(run_solanet_single_apu_nndescent float)
    setup_hip_target(${target_name})

    add_fe_typed_exe(run_solanet_single_apu_nndescent uint8_t)
    setup_hip_target(${target_name})

    add_fe_typed_exe(run_hipvs_single_apu_nndescent float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_exe(run_hipvs_single_apu_nndescent uint8_t)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_ygm_exe(run_solanet_apu float)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})

    add_fe_typed_ygm_exe(run_solanet_apu uint8_t)
    setup_hip_target(${target_name})
    setup_hipvs_target(${target_name})
endif()

if (SALTATLAS_USE_CUVS)
    add_fe_typed_exe(run_cuvs_single_gpu_nndescent float)
    setup_cuvs_target(${target_name})

    add_fe_typed_exe(run_cuvs_single_gpu_nndescent uint8_t)
    setup_cuvs_target(${target_name})

    # SOLANET on a single NVIDIA GPU: the apu_nn kernels compile for CUDA, so
    # the source must be built by the CUDA compiler.
    set_source_files_properties(run_solanet_single_gpu_nndescent.cpp
            PROPERTIES LANGUAGE CUDA)

    add_fe_typed_exe(run_solanet_single_gpu_nndescent float)
    setup_cuvs_target(${target_name})
    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    add_fe_typed_exe(run_solanet_single_gpu_nndescent uint8_t)
    setup_cuvs_target(${target_name})
    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()