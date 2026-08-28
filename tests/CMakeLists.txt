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
