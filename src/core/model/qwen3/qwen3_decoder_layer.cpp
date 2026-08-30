#include "core/model/qwen3/qwen3_decoder_layer.hpp"

#include <algorithm>
#include <utility>

#include "core/kernels/matadd.hpp"

namespace liteinfer::core::model::qwen3
{

namespace
{

// 添加残差连接
std::expected<tensor::Tensor, ModelError>
add_residual(const tensor::Tensor & residual, const tensor::Tensor & branch_output)
{
    const auto residual_shape = residual.shape().values();
    const auto branch_shape = branch_output.shape().values();
    if (residual.data_type() != tensor::DataType::Float32 ||
        branch_output.data_type() != tensor::DataType::Float32 ||
        residual_shape.size() != branch_shape.size() ||
        !std::equal(residual_shape.begin(), residual_shape.end(), branch_shape.begin())) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 DecoderLayer residual tensors must have matching Float32 shapes"
        ));
    }

    auto output = tensor::Tensor::allocate(tensor::DataType::Float32, residual.shape());
    if (!output) {
        return std::unexpected(std::move(output).error());
    }

    auto residual_values = residual.data_as<float>();
    if (!residual_values) {
        return std::unexpected(std::move(residual_values).error());
    }

    auto branch_values = branch_output.data_as<float>();
    if (!branch_values) {
        return std::unexpected(std::move(branch_values).error());
    }

    auto output_values = output->data_as<float>();
    if (!output_values) {
        return std::unexpected(std::move(output_values).error());
    }

    kernels::matadd_f32(*residual_values, *branch_values, *output_values);
    return std::move(*output);
}

} // namespace

Qwen3DecoderLayer::Qwen3DecoderLayer(
    Qwen3Attention self_attn,
    Qwen3MLP mlp,
    layer::RMSNorm input_layernorm,
    layer::RMSNorm post_attention_layernorm
)
    : self_attn_(std::move(self_attn))
    , mlp_(std::move(mlp))
    , input_layernorm_(std::move(input_layernorm))
    , post_attention_layernorm_(std::move(post_attention_layernorm))
{}

std::expected<Qwen3DecoderLayer, ModelError> Qwen3DecoderLayer::create(
    Qwen3Attention self_attn,
    Qwen3MLP mlp,
    layer::RMSNorm input_layernorm,
    layer::RMSNorm post_attention_layernorm
)
{
    const std::size_t hidden_size = self_attn.hidden_size();
    if (mlp.hidden_size() != hidden_size) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 DecoderLayer Attention and MLP hidden sizes must match"
        ));
    }

    if (input_layernorm.normalized_size() != hidden_size ||
        post_attention_layernorm.normalized_size() != hidden_size) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 DecoderLayer RMSNorm dimensions must match the hidden size"
        ));
    }

    if (input_layernorm.eps() != post_attention_layernorm.eps()) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 DecoderLayer RMSNorm layers must use the same epsilon"
        ));
    }

    return Qwen3DecoderLayer(
        std::move(self_attn),
        std::move(mlp),
        std::move(input_layernorm),
        std::move(post_attention_layernorm)
    );
}

std::expected<tensor::Tensor, ModelError> Qwen3DecoderLayer::forward(
    const tensor::Tensor & input
) const
{
    if (input.rank() < 2) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 DecoderLayer input must have shape [..., sequence_length, hidden_size]"
        ));
    }

    auto attention_input = input_layernorm_.forward(input);
    if (!attention_input) {
        return std::unexpected(std::move(attention_input).error());
    }

    auto attention_output = self_attn_.forward(*attention_input);
    if (!attention_output) {
        return std::unexpected(std::move(attention_output).error());
    }

    auto hidden_states = add_residual(input, *attention_output);
    if (!hidden_states) {
        return std::unexpected(std::move(hidden_states).error());
    }

    auto mlp_input = post_attention_layernorm_.forward(*hidden_states);
    if (!mlp_input) {
        return std::unexpected(std::move(mlp_input).error());
    }

    auto mlp_output = mlp_.forward(*mlp_input);
    if (!mlp_output) {
        return std::unexpected(std::move(mlp_output).error());
    }

    return add_residual(*hidden_states, *mlp_output);
}

std::size_t Qwen3DecoderLayer::hidden_size() const noexcept
{
    return self_attn_.hidden_size();
}

std::size_t Qwen3DecoderLayer::intermediate_size() const noexcept
{
    return mlp_.intermediate_size();
}

float Qwen3DecoderLayer::rms_norm_eps() const noexcept
{
    return input_layernorm_.eps();
}

} // namespace liteinfer::core::model::qwen3
