#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "core/common/error.hpp"
#include "core/layer/layer_error.hpp"
#include "core/layer/linear.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_mlp.hpp"
#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core;

tensor::Tensor make_float_tensor(tensor::Shape shape, std::initializer_list<float> values)
{
    auto result = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Float32,
        std::move(shape)
    );
    assert(result.has_value());

    auto data = result->data_as<float>();
    assert(data.has_value());
    assert(data->size() == values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*data)[i] = values.begin()[i];
    }

    return std::move(*result);
}

layer::Linear make_linear(tensor::Shape shape, std::initializer_list<float> values)
{
    auto linear = layer::Linear::create(make_float_tensor(std::move(shape), values));
    assert(linear.has_value());
    return std::move(*linear);
}

layer::Linear make_biased_linear(
    tensor::Shape weight_shape,
    std::initializer_list<float> weight_values,
    tensor::Shape bias_shape,
    std::initializer_list<float> bias_values
)
{
    auto linear = layer::Linear::create(
        make_float_tensor(std::move(weight_shape), weight_values),
        make_float_tensor(std::move(bias_shape), bias_values)
    );
    assert(linear.has_value());
    return std::move(*linear);
}

model::qwen3::Qwen3MLP make_mlp()
{
    auto gate = make_linear(tensor::Shape {3, 2}, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, -1.0F});
    auto up = make_linear(tensor::Shape {3, 2}, {2.0F, 0.0F, 0.0F, 3.0F, 1.0F, 1.0F});
    auto down = make_linear(tensor::Shape {2, 3}, {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, -1.0F});

    auto mlp = model::qwen3::Qwen3MLP::create(std::move(gate), std::move(up), std::move(down));
    assert(mlp.has_value());
    return std::move(*mlp);
}

void assert_near(float actual, float expected, float tolerance = 1.0e-5F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_create_and_accessors()
{
    auto mlp = make_mlp();
    assert(mlp.hidden_size() == 2);
    assert(mlp.intermediate_size() == 3);
}

void test_forward()
{
    auto mlp = make_mlp();
    auto input = make_float_tensor(tensor::Shape {2, 2}, {1.0F, 2.0F, -1.0F, 0.5F});

    auto output = mlp.forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 2);
    assert(*output->shape().extent(0) == 2);
    assert(*output->shape().extent(1) == 2);

    auto values = output->data_as<float>();
    assert(values.has_value());
    assert_near((*values)[0], 0.6552929F);
    assert_near((*values)[1], 11.376389F);
    assert_near((*values)[2], 0.6747020F);
    assert_near((*values)[3], 0.3300254F);
}

void test_forward_preserves_leading_dimensions()
{
    auto mlp = make_mlp();
    auto input = make_float_tensor(tensor::Shape {1, 2, 2}, {1.0F, 2.0F, -1.0F, 0.5F});

    auto output = mlp.forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 1);
    assert(*output->shape().extent(1) == 2);
    assert(*output->shape().extent(2) == 2);
}

void test_empty_input()
{
    auto mlp = make_mlp();
    auto input = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Float32,
        tensor::Shape {0, 2}
    );
    assert(input.has_value());

    auto output = mlp.forward(*input);
    assert(output.has_value());
    assert(output->empty());
    assert(*output->shape().extent(0) == 0);
    assert(*output->shape().extent(1) == 2);
}

void test_create_validation()
{
    auto gate = make_linear(tensor::Shape {3, 2}, {1, 0, 0, 1, 1, -1});
    auto wrong_up = make_linear(tensor::Shape {4, 2}, {1, 0, 0, 1, 1, -1, 1, 1});
    auto down = make_linear(tensor::Shape {2, 3}, {1, 0, 1, 0, 1, -1});

    auto mismatched_branches =
        model::qwen3::Qwen3MLP::create(std::move(gate), std::move(wrong_up), std::move(down));
    assert(!mismatched_branches.has_value());
    assert(mismatched_branches.error().category() == common::ErrorCategory::Model);
    assert(
        mismatched_branches.error().code() ==
        std::to_underlying(model::ModelErrorCode::InvalidConfiguration)
    );

    gate = make_linear(tensor::Shape {3, 2}, {1, 0, 0, 1, 1, -1});
    auto up = make_linear(tensor::Shape {3, 2}, {1, 0, 0, 1, 1, -1});
    auto wrong_down = make_linear(tensor::Shape {2, 4}, {1, 0, 1, 0, 0, 1, -1, 0});
    auto mismatched_down =
        model::qwen3::Qwen3MLP::create(std::move(gate), std::move(up), std::move(wrong_down));
    assert(!mismatched_down.has_value());
    assert(mismatched_down.error().category() == common::ErrorCategory::Model);

    auto biased_gate =
        make_biased_linear(tensor::Shape {3, 2}, {1, 0, 0, 1, 1, -1}, tensor::Shape {3}, {0, 0, 0});
    up = make_linear(tensor::Shape {3, 2}, {1, 0, 0, 1, 1, -1});
    down = make_linear(tensor::Shape {2, 3}, {1, 0, 1, 0, 1, -1});
    auto biased_projection =
        model::qwen3::Qwen3MLP::create(std::move(biased_gate), std::move(up), std::move(down));
    assert(!biased_projection.has_value());
    assert(biased_projection.error().category() == common::ErrorCategory::Model);
}

void test_forward_propagates_layer_error()
{
    auto mlp = make_mlp();
    auto input = make_float_tensor(tensor::Shape {1, 3}, {1.0F, 2.0F, 3.0F});

    auto output = mlp.forward(input);
    assert(!output.has_value());
    assert(output.error().category() == common::ErrorCategory::Layer);
    assert(output.error().code() == std::to_underlying(layer::LayerErrorCode::InvalidInput));
}

} // namespace

int main()
{
    test_create_and_accessors();
    test_forward();
    test_forward_preserves_leading_dimensions();
    test_empty_input();
    test_create_validation();
    test_forward_propagates_layer_error();
}
