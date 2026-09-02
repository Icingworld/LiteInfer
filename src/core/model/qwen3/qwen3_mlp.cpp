#include "core/model/qwen3/qwen3_mlp.hpp"

#include <utility>

#include "core/kernels/silu_mul.hpp"

namespace liteinfer::core::model::qwen3
{

Qwen3MLP::Qwen3MLP(layer::Linear gate_proj, layer::Linear up_proj, layer::Linear down_proj)
    : gate_proj_(std::move(gate_proj))
    , up_proj_(std::move(up_proj))
    , down_proj_(std::move(down_proj))
{}

std::expected<Qwen3MLP, ModelError>
Qwen3MLP::create(layer::Linear gate_proj, layer::Linear up_proj, layer::Linear down_proj)
{
    const std::size_t hidden_size = gate_proj.in_features();
    const std::size_t intermediate_size = gate_proj.out_features();

    if (up_proj.in_features() != hidden_size || up_proj.out_features() != intermediate_size) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 MLP gate and up projections must have matching dimensions"
        ));
    }

    if (down_proj.in_features() != intermediate_size || down_proj.out_features() != hidden_size) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 MLP down projection dimensions must reverse the gate projection"
        ));
    }

    if (gate_proj.has_bias() || up_proj.has_bias() || down_proj.has_bias()) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 MLP projections must not contain bias"
        ));
    }

    return Qwen3MLP(std::move(gate_proj), std::move(up_proj), std::move(down_proj));
}

std::expected<tensor::Tensor, ModelError> Qwen3MLP::forward(const tensor::Tensor & input) const
{
    auto gate = gate_proj_.forward(input);
    if (!gate) [[unlikely]] {
        return std::unexpected(std::move(gate).error());
    }

    auto up = up_proj_.forward(input);
    if (!up) [[unlikely]] {
        return std::unexpected(std::move(up).error());
    }

    auto intermediate =
        tensor::Tensor::allocate(common::data_type::DataType::Float32, gate->shape());
    if (!intermediate) [[unlikely]] {
        return std::unexpected(std::move(intermediate).error());
    }

    auto gate_values = gate->data_as<float>();
    if (!gate_values) [[unlikely]] {
        return std::unexpected(std::move(gate_values).error());
    }

    auto up_values = up->data_as<float>();
    if (!up_values) [[unlikely]] {
        return std::unexpected(std::move(up_values).error());
    }

    auto intermediate_values = intermediate->data_as<float>();
    if (!intermediate_values) [[unlikely]] {
        return std::unexpected(std::move(intermediate_values).error());
    }

    kernels::silu_mul_f32(*gate_values, *up_values, *intermediate_values);
    return down_proj_.forward(*intermediate);
}

std::size_t Qwen3MLP::hidden_size() const noexcept
{
    return gate_proj_.in_features();
}

std::size_t Qwen3MLP::intermediate_size() const noexcept
{
    return gate_proj_.out_features();
}

} // namespace liteinfer::core::model::qwen3
