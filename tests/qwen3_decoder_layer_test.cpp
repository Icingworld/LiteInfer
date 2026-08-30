#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>

#include "core/common/error.hpp"
#include "core/layer/layer_error.hpp"
#include "core/layer/linear.hpp"
#include "core/layer/rms_norm.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_attention.hpp"
#include "core/model/qwen3/qwen3_decoder_layer.hpp"
#include "core/model/qwen3/qwen3_mlp.hpp"
#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core;

tensor::Tensor make_float_tensor(tensor::Shape shape, std::initializer_list<float> values)
{
    auto result = tensor::Tensor::allocate(tensor::DataType::Float32, std::move(shape));
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

layer::RMSNorm make_norm(std::initializer_list<float> weights, float eps = 1.0F)
{
    auto norm =
        layer::RMSNorm::create(make_float_tensor(tensor::Shape {weights.size()}, weights), eps);
    assert(norm.has_value());
    return std::move(*norm);
}

model::qwen3::Qwen3Attention make_attention()
{
    auto attention = model::qwen3::Qwen3Attention::create(
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_norm({1.0F, 1.0F}),
        make_norm({1.0F, 1.0F}),
        1,
        1,
        10'000.0F
    );
    assert(attention.has_value());
    return std::move(*attention);
}

model::qwen3::Qwen3MLP make_mlp()
{
    auto mlp = model::qwen3::Qwen3MLP::create(
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}),
        make_linear(tensor::Shape {2, 2}, {1.0F, 0.0F, 0.0F, 1.0F})
    );
    assert(mlp.has_value());
    return std::move(*mlp);
}

model::qwen3::Qwen3MLP make_mismatched_mlp()
{
    const std::initializer_list<float> identity = {
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
    };
    auto mlp = model::qwen3::Qwen3MLP::create(
        make_linear(tensor::Shape {3, 3}, identity),
        make_linear(tensor::Shape {3, 3}, identity),
        make_linear(tensor::Shape {3, 3}, identity)
    );
    assert(mlp.has_value());
    return std::move(*mlp);
}

std::expected<model::qwen3::Qwen3DecoderLayer, model::ModelError> make_decoder_layer_result(
    float post_attention_eps = 1.0F
)
{
    return model::qwen3::Qwen3DecoderLayer::create(
        make_attention(),
        make_mlp(),
        make_norm({1.0F, 2.0F}),
        make_norm({0.5F, 1.5F}, post_attention_eps)
    );
}

model::qwen3::Qwen3DecoderLayer make_decoder_layer()
{
    auto decoder_layer = make_decoder_layer_result();
    assert(decoder_layer.has_value());
    return std::move(*decoder_layer);
}

void assert_near(float actual, float expected, float tolerance = 1.0e-5F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_create_and_accessors()
{
    auto decoder_layer = make_decoder_layer();
    assert(decoder_layer.hidden_size() == 2);
    assert(decoder_layer.intermediate_size() == 2);
    assert(decoder_layer.rms_norm_eps() == 1.0F);
}

void test_forward_pre_norm_attention_mlp_and_residuals()
{
    auto decoder_layer = make_decoder_layer();
    auto input = make_float_tensor(tensor::Shape {1, 1, 2}, {1.0F, 2.0F});

    auto output = decoder_layer.forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 1);
    assert(*output->shape().extent(1) == 1);
    assert(*output->shape().extent(2) == 2);

    auto values = output->data_as<float>();
    assert(values.has_value());
    assert_near((*values)[0], 1.5651248F);
    assert_near((*values)[1], 7.2565380F);
}

void test_empty_sequence()
{
    auto decoder_layer = make_decoder_layer();
    auto input = tensor::Tensor::allocate(tensor::DataType::Float32, tensor::Shape {1, 0, 2});
    assert(input.has_value());

    auto output = decoder_layer.forward(*input);
    assert(output.has_value());
    assert(output->empty());
    assert(output->shape().values()[0] == 1);
    assert(output->shape().values()[1] == 0);
    assert(output->shape().values()[2] == 2);
}

void test_create_validation()
{
    auto mismatched_mlp = model::qwen3::Qwen3DecoderLayer::create(
        make_attention(),
        make_mismatched_mlp(),
        make_norm({1.0F, 1.0F}),
        make_norm({1.0F, 1.0F})
    );
    assert(!mismatched_mlp.has_value());
    assert(mismatched_mlp.error().category() == common::ErrorCategory::Model);
    assert(
        mismatched_mlp.error().code() ==
        std::to_underlying(model::ModelErrorCode::InvalidConfiguration)
    );

    auto mismatched_norm = model::qwen3::Qwen3DecoderLayer::create(
        make_attention(),
        make_mlp(),
        make_norm({1.0F, 1.0F, 1.0F}),
        make_norm({1.0F, 1.0F})
    );
    assert(!mismatched_norm.has_value());
    assert(mismatched_norm.error().category() == common::ErrorCategory::Model);

    auto mismatched_eps = make_decoder_layer_result(1.0e-6F);
    assert(!mismatched_eps.has_value());
    assert(mismatched_eps.error().category() == common::ErrorCategory::Model);
}

void test_forward_validation_and_error_propagation()
{
    auto decoder_layer = make_decoder_layer();
    auto rank_one_input = make_float_tensor(tensor::Shape {2}, {1.0F, 2.0F});

    auto rank_one_output = decoder_layer.forward(rank_one_input);
    assert(!rank_one_output.has_value());
    assert(rank_one_output.error().category() == common::ErrorCategory::Model);
    assert(
        rank_one_output.error().code() == std::to_underlying(model::ModelErrorCode::InvalidInput)
    );

    auto wrong_hidden_size = make_float_tensor(tensor::Shape {1, 1, 3}, {1.0F, 2.0F, 3.0F});
    auto wrong_hidden_output = decoder_layer.forward(wrong_hidden_size);
    assert(!wrong_hidden_output.has_value());
    assert(wrong_hidden_output.error().category() == common::ErrorCategory::Layer);
    assert(
        wrong_hidden_output.error().code() ==
        std::to_underlying(layer::LayerErrorCode::InvalidInput)
    );
}

} // namespace

int main()
{
    test_create_and_accessors();
    test_forward_pre_norm_attention_mlp_and_residuals();
    test_empty_sequence();
    test_create_validation();
    test_forward_validation_and_error_propagation();
}
