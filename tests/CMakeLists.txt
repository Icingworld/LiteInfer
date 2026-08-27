add_executable(liteinfer_tensor_test
    tensor_test.cpp
)

target_link_libraries(liteinfer_tensor_test
    PRIVATE
        liteinfer_core_tensor
)

add_test(NAME liteinfer_tensor_test COMMAND liteinfer_tensor_test)
