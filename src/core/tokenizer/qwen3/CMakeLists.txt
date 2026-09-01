set(LITEINFER_CORE_TOKENIZER_QWEN3_SOURCES
    qwen3_tokenizer.cpp
)

add_library(liteinfer_core_tokenizer_qwen3 STATIC
    ${LITEINFER_CORE_TOKENIZER_QWEN3_SOURCES}
)

target_include_directories(liteinfer_core_tokenizer_qwen3
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../
)

target_link_libraries(liteinfer_core_tokenizer_qwen3
    PUBLIC
        liteinfer_core_tokenizer
        liteinfer_core_filesystem
    PRIVATE
        liteinfer_core_io
        liteinfer_core_json
)
