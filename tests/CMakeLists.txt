add_executable(liteinfer_tensor_test
    tensor_test.cpp
)

target_link_libraries(liteinfer_tensor_test
    PRIVATE
        liteinfer_core_tensor
)

add_test(NAME liteinfer_tensor_test COMMAND liteinfer_tensor_test)

add_executable(liteinfer_embedding_test
    embedding_test.cpp
)

target_link_libraries(liteinfer_embedding_test
    PRIVATE
        liteinfer_core_embedding
)

add_test(NAME liteinfer_embedding_test COMMAND liteinfer_embedding_test)

add_executable(liteinfer_kernels_test
    kernels_test.cpp
)

target_link_libraries(liteinfer_kernels_test
    PRIVATE
        liteinfer_core_kernels
)

add_test(NAME liteinfer_kernels_test COMMAND liteinfer_kernels_test)

add_executable(liteinfer_layer_test
    linear_test.cpp
)

target_link_libraries(liteinfer_layer_test
    PRIVATE
        liteinfer_core_layer
)

add_test(NAME liteinfer_layer_test COMMAND liteinfer_layer_test)

add_executable(liteinfer_rms_norm_test
    rms_norm_test.cpp
)

target_link_libraries(liteinfer_rms_norm_test
    PRIVATE
        liteinfer_core_layer
)

add_test(NAME liteinfer_rms_norm_test COMMAND liteinfer_rms_norm_test)

add_executable(liteinfer_qwen3_mlp_test
    qwen3_mlp_test.cpp
)

target_link_libraries(liteinfer_qwen3_mlp_test
    PRIVATE
        liteinfer_core_model_qwen3
)

add_test(NAME liteinfer_qwen3_mlp_test COMMAND liteinfer_qwen3_mlp_test)

add_executable(liteinfer_qwen3_attention_test
    qwen3_attention_test.cpp
)

target_link_libraries(liteinfer_qwen3_attention_test
    PRIVATE
        liteinfer_core_model_qwen3
)

add_test(NAME liteinfer_qwen3_attention_test COMMAND liteinfer_qwen3_attention_test)

add_executable(liteinfer_qwen3_decoder_layer_test
    qwen3_decoder_layer_test.cpp
)

target_link_libraries(liteinfer_qwen3_decoder_layer_test
    PRIVATE
        liteinfer_core_model_qwen3
)

add_test(NAME liteinfer_qwen3_decoder_layer_test COMMAND liteinfer_qwen3_decoder_layer_test)

if(UNIX)
    add_executable(liteinfer_filesystem_test
        filesystem_test.cpp
    )

    target_link_libraries(liteinfer_filesystem_test
        PRIVATE
            liteinfer_core_filesystem
    )

    add_test(NAME liteinfer_filesystem_test COMMAND liteinfer_filesystem_test)
endif()
