#include "core/layer/rms_norm.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "core/kernels/rms_norm.hpp"

namespace liteinfer::core::layer
{

RMSNorm::RMSNorm(tensor::Tensor weight, float eps)
    : weight_(std::move(weight))
    , eps_(eps)
{}

std::expected<RMSNorm, LayerError> RMSNorm::create(tensor::Tensor weight, float eps)
{
    if (weight.rank() != 1) [[unlikely]] {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "RMSNorm weight must be a 1D tensor")
        );
    }

    if (*weight.shape().extent(0) == 0) [[unlikely]] {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "RMSNorm weight must be non-empty")
        );
    }

    if (weight.data_type() != common::data_type::DataType::Float32) [[unlikely]] {
        return std::unexpected(LayerError(
            LayerErrorCode::UnsupportedDataType,
            "RMSNorm weight must use Float32 data type"
        ));
    }

    if (!weight.is_contiguous()) [[unlikely]] {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "RMSNorm weight must be contiguous")
        );
    }

    if (!std::isfinite(eps) || eps <= 0.0F) [[unlikely]] {
        return std::unexpected(LayerError(
            LayerErrorCode::InvalidEpsilon,
            "RMSNorm epsilon must be finite and greater than zero"
        ));
    }

    return RMSNorm(std::move(weight), eps);
}

std::expected<tensor::Tensor, LayerError> RMSNorm::forward(const tensor::Tensor & input) const
{
    if (input.rank() == 0) [[unlikely]] {
        return std::unexpected(LayerError(
            LayerErrorCode::InvalidInput,
            "RMSNorm input must have at least one dimension"
        ));
    }

    if (!input.is_contiguous()) [[unlikely]] {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidInput, "RMSNorm input must be contiguous")
        );
    }

    if (input.data_type() != common::data_type::DataType::Float32) [[unlikely]] {
        return std::unexpected(LayerError(
            LayerErrorCode::UnsupportedDataType,
            "RMSNorm input must use Float32 data type"
        ));
    }

    const auto input_shape = input.shape().values();
    if (input_shape.back() != normalized_size()) [[unlikely]] {
        return std::unexpected(LayerError(
            LayerErrorCode::InvalidInput,
            "RMSNorm input normalized dimension does not match the weight"
        ));
    }

    std::vector<std::size_t> output_dimensions(input_shape.begin(), input_shape.end());
    auto output = tensor::Tensor::allocate(
        common::data_type::DataType::Float32,
        tensor::Shape(std::move(output_dimensions))
    );
    if (!output) [[unlikely]] {
        return std::unexpected(std::move(output).error());
    }

    auto input_values = input.data_as<float>();
    if (!input_values) [[unlikely]] {
        return std::unexpected(std::move(input_values).error());
    }

    auto weight_values = weight_.data_as<float>();
    if (!weight_values) [[unlikely]] {
        return std::unexpected(std::move(weight_values).error());
    }

    auto output_values = output->data_as<float>();
    if (!output_values) [[unlikely]] {
        return std::unexpected(std::move(output_values).error());
    }

    if (!input.empty()) {
        kernels::rms_norm_f32(
            *input_values,
            *weight_values,
            *output_values,
            normalized_size(),
            eps_
        );
    }

    return std::move(*output);
}

std::size_t RMSNorm::normalized_size() const noexcept
{
    // 创建 RMSNorm 时，权重张量的秩和大小已经验证，这里认为一定不会越界
    return *weight_.shape().extent(0);
}

float RMSNorm::eps() const noexcept
{
    return eps_;
}

} // namespace liteinfer::core::layer
