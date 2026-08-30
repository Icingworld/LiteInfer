#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <utility>

#include "core/common/error.hpp"
#include "core/layer/layer_error.hpp"
#include "core/layer/linear.hpp"
#include "core/layer/rms_norm.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_attention.hpp"
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

layer::Linear
make_linear(tensor::Shape shape, std::initializer_list<float> values, bool with_bias = false)
{
    auto weight = make_float_tensor(std::move(shape), values);
    std::optional<tensor::Tensor> bias;
    if (with_bias) {
        auto bias_result = tensor::Tensor::allocate(
            tensor::DataType::Float32,
            tensor::Shape {*weight.shape().extent(0)}
        );
        assert(bias_result.has_value());
        bias = std::move(*bias_result);
    }

    auto linear = layer::Linear::create(std::move(weight), std::move(bias));
    assert(linear.has_value());
    return std::move(*linear);
}

layer::RMSNorm make_norm(std::size_t size, float eps = 1.0e-6F)
{
    auto weight = tensor::Tensor::allocate(tensor::DataType::Float32, tensor::Shape {size});
    assert(weight.has_value());

    auto values = weight->data_as<float>();
    assert(values.has_value());
    for (float & value : *values) {
        value = 1.0F;
    }

    auto norm = layer::RMSNorm::create(std::move(*weight), eps);
    assert(norm.has_value());
    return std::move(*norm);
}

std::expected<model::qwen3::Qwen3Attention, model::ModelError> make_attention_result(
    std::size_t num_attention_heads = 2,
    std::size_t num_key_value_heads = 1,
    float rope_theta = 10'000.0F,
    std::size_t key_norm_size = 2,
    bool q_bias = false,
    bool other_bias = false
)
{
    auto q_proj = make_linear(
        tensor::Shape {4, 4},
        {
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        },
        q_bias
    );
    auto k_proj = make_linear(
        tensor::Shape {2, 4},
        {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F},
        other_bias
    );
    auto v_proj = make_linear(
        tensor::Shape {2, 4},
        {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F},
        other_bias
    );
    auto o_proj = make_linear(
        tensor::Shape {4, 4},
        {
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        },
        other_bias
    );

    return model::qwen3::Qwen3Attention::create(
        std::move(q_proj),
        std::move(k_proj),
        std::move(v_proj),
        std::move(o_proj),
        make_norm(2),
        make_norm(key_norm_size),
        num_attention_heads,
        num_key_value_heads,
        rope_theta
    );
}

model::qwen3::Qwen3Attention make_attention()
{
    auto attention = make_attention_result();
    assert(attention.has_value());
    return std::move(*attention);
}

void assert_near(float actual, float expected, float tolerance = 1.0e-4F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void assert_reference_output(const tensor::Tensor & output, std::size_t offset)
{
    auto values = output.data_as<float>();
    assert(values.has_value());

    const float expected[] = {
        10.0F,
        20.0F,
        10.0F,
        20.0F,
        28.622664F,
        38.622665F,
        26.709459F,
        36.709461F,
    };
    for (std::size_t i = 0; i < 8; ++i) {
        assert_near((*values)[offset + i], expected[i]);
    }
}

void test_create_and_accessors()
{
    auto attention = make_attention();
    assert(attention.hidden_size() == 4);
    assert(attention.num_attention_heads() == 2);
    assert(attention.num_key_value_heads() == 1);
    assert(attention.head_dim() == 2);
    assert(attention.rope_theta() == 10'000.0F);
}

void test_forward_gqa_rope_and_causal_mask()
{
    auto attention = make_attention();
    auto input = make_float_tensor(
        tensor::Shape {1, 2, 4},
        {1.0F, 0.0F, 10.0F, 20.0F, 0.0F, 1.0F, 30.0F, 40.0F}
    );

    auto output = attention.forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 1);
    assert(*output->shape().extent(1) == 2);
    assert(*output->shape().extent(2) == 4);
    assert_reference_output(*output, 0);
}

void test_batches_are_isolated()
{
    auto attention = make_attention();
    auto input = make_float_tensor(
        tensor::Shape {2, 2, 4},
        {
            1.0F,
            0.0F,
            10.0F,
            20.0F,
            0.0F,
            1.0F,
            30.0F,
            40.0F,
            1.0F,
            0.0F,
            10.0F,
            20.0F,
            0.0F,
            1.0F,
            30.0F,
            40.0F,
        }
    );

    auto output = attention.forward(input);
    assert(output.has_value());
    assert_reference_output(*output, 0);
    assert_reference_output(*output, 8);
}

void test_empty_sequence()
{
    auto attention = make_attention();
    auto input = tensor::Tensor::allocate(tensor::DataType::Float32, tensor::Shape {1, 0, 4});
    assert(input.has_value());

    auto output = attention.forward(*input);
    assert(output.has_value());
    assert(output->empty());
    assert(*output->shape().extent(0) == 1);
    assert(*output->shape().extent(1) == 0);
    assert(*output->shape().extent(2) == 4);
}

void test_create_validation()
{
    auto zero_heads = make_attention_result(0, 1);
    assert(!zero_heads.has_value());
    assert(zero_heads.error().category() == common::ErrorCategory::Model);
    assert(
        zero_heads.error().code() == std::to_underlying(model::ModelErrorCode::InvalidConfiguration)
    );

    auto invalid_gqa = make_attention_result(3, 2);
    assert(!invalid_gqa.has_value());
    assert(invalid_gqa.error().category() == common::ErrorCategory::Model);

    auto mismatched_norm = make_attention_result(2, 1, 10'000.0F, 4);
    assert(!mismatched_norm.has_value());
    assert(mismatched_norm.error().category() == common::ErrorCategory::Model);

    auto invalid_rope = make_attention_result(2, 1, 0.0F);
    assert(!invalid_rope.has_value());
    assert(invalid_rope.error().category() == common::ErrorCategory::Model);

    auto inconsistent_bias = make_attention_result(2, 1, 10'000.0F, 2, true, false);
    assert(!inconsistent_bias.has_value());
    assert(inconsistent_bias.error().category() == common::ErrorCategory::Model);
}

void test_forward_validation()
{
    auto attention = make_attention();

    auto rank_one = make_float_tensor(tensor::Shape {4}, {1.0F, 2.0F, 3.0F, 4.0F});
    auto rank_one_output = attention.forward(rank_one);
    assert(!rank_one_output.has_value());
    assert(rank_one_output.error().category() == common::ErrorCategory::Model);
    assert(
        rank_one_output.error().code() == std::to_underlying(model::ModelErrorCode::InvalidInput)
    );

    auto wrong_hidden_size =
        make_float_tensor(tensor::Shape {1, 2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto wrong_hidden_size_output = attention.forward(wrong_hidden_size);
    assert(!wrong_hidden_size_output.has_value());
    assert(wrong_hidden_size_output.error().category() == common::ErrorCategory::Layer);
    assert(
        wrong_hidden_size_output.error().code() ==
        std::to_underlying(layer::LayerErrorCode::InvalidInput)
    );
}

} // namespace

int main()
{
    test_create_and_accessors();
    test_forward_gqa_rope_and_causal_mask();
    test_batches_are_isolated();
    test_empty_sequence();
    test_create_validation();
    test_forward_validation();
}
