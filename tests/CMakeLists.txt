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
