set(LITEINFER_CORE_MODEL_QWEN3_SOURCES
    qwen3_mlp.cpp
)

add_library(liteinfer_core_model_qwen3 STATIC ${LITEINFER_CORE_MODEL_QWEN3_SOURCES})

target_include_directories(liteinfer_core_model_qwen3
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../
)

target_link_libraries(liteinfer_core_model_qwen3
    PUBLIC
        liteinfer_core_layer
    PRIVATE
        liteinfer_core_kernels
)
