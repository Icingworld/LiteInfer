set(LITEINFER_CORE_TENSOR_SOURCES
    shape.cpp
    strides.cpp
    tensor.cpp
    tensor_view.cpp
)

add_library(liteinfer_core_tensor STATIC ${LITEINFER_CORE_TENSOR_SOURCES})

target_link_libraries(liteinfer_core_tensor
    PUBLIC
        liteinfer_core_common
)

target_include_directories(liteinfer_core_tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../
)
