set(LITEINFER_CORE_KERNELS_SOURCES
    rms_norm.cpp
)

add_library(liteinfer_core_kernels STATIC ${LITEINFER_CORE_KERNELS_SOURCES})

target_include_directories(liteinfer_core_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_kernels
    PUBLIC
        liteinfer_core_tensor
        liteinfer_core_embedding
)
