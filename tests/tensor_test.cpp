#include <cassert>
#include <cstdint>
#include <utility>

#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core::tensor;

void test_float32_access()
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, Shape {2, 3});
    assert(result.has_value());

    Tensor tensor = std::move(*result);
    assert(tensor.rank() == 2);
    assert(tensor.numel() == 6);
    assert(tensor.element_size() == sizeof(float));
    assert(tensor.data().size() == 6 * sizeof(float));
    assert(tensor.is_contiguous());
    assert(tensor.numel() != 0);

    auto values = tensor.data_as<float>();
    assert(values.has_value());
    assert(values->size() == 6);
    (*values)[0] = 1.5F;

    const Tensor & const_tensor = tensor;
    auto read_only_values = const_tensor.data_as<float>();
    assert(read_only_values.has_value());
    assert((*read_only_values)[0] == 1.5F);
}

void test_data_type_mismatch()
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, Shape {2});
    assert(result.has_value());

    const Tensor & tensor = *result;
    auto values = tensor.data_as<std::int32_t>();
    assert(!values.has_value());
    assert(values.error().code() == std::to_underlying(TensorErrorCode::DataTypeMismatch));
}

void test_unsupported_cpp_type()
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, Shape {2});
    assert(result.has_value());

    const Tensor & tensor = *result;
    auto values = tensor.data_as<double>();
    assert(!values.has_value());
    assert(values.error().code() == std::to_underlying(TensorErrorCode::InvalidDataType));
}

void test_zero_element_tensor()
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Int32, Shape {0, 4});
    assert(result.has_value());

    Tensor tensor = std::move(*result);
    assert(tensor.numel() == 0);

    auto values = tensor.data_as<std::int32_t>();
    assert(values.has_value());
    assert(values->empty());
}

} // namespace

int main()
{
    test_float32_access();
    test_data_type_mismatch();
    test_unsupported_cpp_type();
    test_zero_element_tensor();
}
