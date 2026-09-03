#include <cassert>
#include <cstdint>
#include <utility>

#include "core/layer/layer_error.hpp"
#include "core/layer/linear.hpp"

namespace
{

using namespace liteinfer::core::layer;
using namespace liteinfer::core::tensor;

Tensor make_float_tensor(Shape shape, std::initializer_list<float> values)
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, std::move(shape));
    assert(result.has_value());

    auto data = result->data_as<float>();
    assert(data.has_value());
    assert(data->size() == values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*data)[i] = values.begin()[i];
    }

    return std::move(*result);
}

void assert_float_values(const Tensor & tensor, std::initializer_list<float> expected)
{
    auto values = tensor.data_as<float>();
    assert(values.has_value());
    assert(values->size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert((*values)[i] == expected.begin()[i]);
    }
}

void test_forward_with_bias()
{
    auto weight = make_float_tensor(Shape {3, 2}, {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F});
    auto bias = make_float_tensor(Shape {3}, {1.0F, 2.0F, 3.0F});
    auto input = make_float_tensor(Shape {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});

    auto linear = Linear::create(std::move(weight), std::move(bias));
    assert(linear.has_value());
    assert(linear->in_features() == 2);
    assert(linear->out_features() == 3);

    auto output = linear->forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 2);
    assert(*output->shape().extent(0) == 2);
    assert(*output->shape().extent(1) == 3);
    assert_float_values(*output, {51.0F, 112.0F, 173.0F, 111.0F, 252.0F, 393.0F});
}

void test_forward_preserves_leading_dimensions()
{
    auto weight = make_float_tensor(Shape {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto input = make_float_tensor(
        Shape {2, 2, 3},
        {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F}
    );

    auto linear = Linear::create(std::move(weight));
    assert(linear.has_value());

    auto output = linear->forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 2);
    assert(*output->shape().extent(1) == 2);
    assert(*output->shape().extent(2) == 2);
    assert_float_values(*output, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F, 6.0F, 15.0F});
}

void test_empty_input()
{
    auto weight = make_float_tensor(Shape {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto input_result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, Shape {0, 3});
    assert(input_result.has_value());

    auto linear = Linear::create(std::move(weight));
    assert(linear.has_value());

    auto output = linear->forward(*input_result);
    assert(output.has_value());
    assert(output->numel() == 0);
    assert(output->shape().rank() == 2);
    assert(*output->shape().extent(0) == 0);
    assert(*output->shape().extent(1) == 2);
}

void test_create_validation()
{
    auto rank_one_weight = make_float_tensor(Shape {2}, {1.0F, 2.0F});
    auto rank_one_result = Linear::create(std::move(rank_one_weight));
    assert(!rank_one_result.has_value());
    assert(rank_one_result.error().code() == std::to_underlying(LayerErrorCode::InvalidWeight));

    auto empty_weight = make_float_tensor(Shape {0, 2}, {});
    auto empty_result = Linear::create(std::move(empty_weight));
    assert(!empty_result.has_value());
    assert(empty_result.error().code() == std::to_underlying(LayerErrorCode::InvalidWeight));

    auto integer_weight_result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Int32, Shape {2, 2});
    assert(integer_weight_result.has_value());
    auto integer_result = Linear::create(std::move(*integer_weight_result));
    assert(!integer_result.has_value());
    assert(
        integer_result.error().code() == std::to_underlying(LayerErrorCode::UnsupportedDataType)
    );

    auto weight = make_float_tensor(Shape {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    auto invalid_bias = make_float_tensor(Shape {1}, {1.0F});
    auto invalid_bias_result = Linear::create(std::move(weight), std::move(invalid_bias));
    assert(!invalid_bias_result.has_value());
    assert(invalid_bias_result.error().code() == std::to_underlying(LayerErrorCode::InvalidBias));
}

void test_forward_validation()
{
    auto weight = make_float_tensor(Shape {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    auto linear = Linear::create(std::move(weight));
    assert(linear.has_value());

    auto wrong_shape_input = make_float_tensor(Shape {1, 3}, {1.0F, 2.0F, 3.0F});
    auto wrong_shape_result = linear->forward(wrong_shape_input);
    assert(!wrong_shape_result.has_value());
    assert(wrong_shape_result.error().code() == std::to_underlying(LayerErrorCode::InvalidInput));

    auto integer_input_result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Int32, Shape {1, 2});
    assert(integer_input_result.has_value());
    auto integer_result = linear->forward(*integer_input_result);
    assert(!integer_result.has_value());
    assert(
        integer_result.error().code() == std::to_underlying(LayerErrorCode::UnsupportedDataType)
    );

    auto scalar_input = make_float_tensor(Shape {}, {1.0F});
    auto scalar_result = linear->forward(scalar_input);
    assert(!scalar_result.has_value());
    assert(scalar_result.error().code() == std::to_underlying(LayerErrorCode::InvalidInput));
}

} // namespace

int main()
{
    test_forward_with_bias();
    test_forward_preserves_leading_dimensions();
    test_empty_input();
    test_create_validation();
    test_forward_validation();
}
