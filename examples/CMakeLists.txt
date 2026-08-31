add_executable(liteinfer_qwen3_forward
    qwen3_forward.cpp
)

target_link_libraries(liteinfer_qwen3_forward
    PRIVATE
        liteinfer_core_model_qwen3
)
