set(LITEINFER_CORE_LAYER_SOURCES
    linear.cpp
)

add_library(liteinfer_core_layer STATIC ${LITEINFER_CORE_LAYER_SOURCES})

target_include_directories(liteinfer_core_layer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(liteinfer_core_layer
    PUBLIC
        liteinfer_core_tensor
    PRIVATE
        liteinfer_core_kernels
)
