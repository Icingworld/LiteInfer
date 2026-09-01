add_executable(liteinfer_qwen3_forward
    qwen3_forward.cpp
)

target_link_libraries(liteinfer_qwen3_forward
    PRIVATE
        liteinfer_core_model_qwen3
)

add_executable(liteinfer_qwen3_tokenizer
    qwen3_tokenizer.cpp
)

target_link_libraries(liteinfer_qwen3_tokenizer
    PRIVATE
        liteinfer_core_tokenizer_qwen3
)
