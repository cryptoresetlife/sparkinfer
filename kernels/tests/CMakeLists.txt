# CPU reference correctness tests — pure C++, no CUDA needed. They validate the
# kernel algorithms against an independent double-precision ground truth.
add_executable(cpu_reference_test cpu_reference_test.cpp)
set_target_properties(cpu_reference_test PROPERTIES CXX_STANDARD 17)
add_test(NAME cpu_reference_test COMMAND cpu_reference_test)

add_executable(nvfp4_dp4a_decode_gpu_test nvfp4_dp4a_decode_gpu_test.cu)
target_link_libraries(nvfp4_dp4a_decode_gpu_test PRIVATE si_gemm si_fused si_quant CUDA::cudart)
set_target_properties(nvfp4_dp4a_decode_gpu_test PROPERTIES CUDA_STANDARD 17)
add_test(NAME nvfp4_dp4a_decode_gpu_test COMMAND nvfp4_dp4a_decode_gpu_test)
set_tests_properties(nvfp4_dp4a_decode_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME nvfp4_dp4a_decode_legacy_gpu_test
  COMMAND ${CMAKE_COMMAND} -E env SPARKINFER_NVFP4_ROWS_PAIRWISE=0
          $<TARGET_FILE:nvfp4_dp4a_decode_gpu_test>)
set_tests_properties(nvfp4_dp4a_decode_legacy_gpu_test PROPERTIES SKIP_RETURN_CODE 77)

if(BUILD_NVFP4_KERNELS)
  add_executable(prefill_nvfp4_gpu_test prefill_nvfp4_gpu_test.cpp)
  # si_nvfp4, not si_fused: the FP4 entry points live in their own whole-program library so that
  # the sm_120a device code never has to device-link against the rest of the build.
  target_link_libraries(prefill_nvfp4_gpu_test PRIVATE si_nvfp4 si_fused si_quant CUDA::cudart)
  set_target_properties(prefill_nvfp4_gpu_test PROPERTIES CUDA_ARCHITECTURES "120a")
  add_test(NAME prefill_nvfp4_gpu_test COMMAND prefill_nvfp4_gpu_test)
  set_tests_properties(prefill_nvfp4_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
