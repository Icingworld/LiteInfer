add_library(liteinfer_core_runtime_qwen3 STATIC
    qwen3_text_generator.cpp
)

target_include_directories(liteinfer_core_runtime_qwen3
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../
)

target_link_libraries(liteinfer_core_runtime_qwen3
    PUBLIC
        liteinfer_core_model_qwen3
        liteinfer_core_tokenizer_qwen3
)
