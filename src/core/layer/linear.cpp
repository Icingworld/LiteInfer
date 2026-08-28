#include "core/layer/linear.hpp"

#include <utility>
#include <vector>

#include "core/kernels/matmul.hpp"

namespace liteinfer::core::layer
{

Linear::Linear(tensor::Tensor weight, std::optional<tensor::Tensor> bias)
    : weight_(std::move(weight))
    , bias_(std::move(bias))
{}

std::expected<Linear, LayerError>
Linear::create(tensor::Tensor weight, std::optional<tensor::Tensor> bias)
{
    if (weight.rank() != 2) {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "Linear weight must be a 2D tensor")
        );
    }

    const auto weight_shape = weight.shape().values();
    const std::size_t out_features = weight_shape[0];
    const std::size_t in_features = weight_shape[1];
    if (out_features == 0 || in_features == 0) {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "Linear weight must be non-empty")
        );
    }

    if (weight.data_type() != tensor::DataType::Float32) {
        return std::unexpected(LayerError(
            LayerErrorCode::UnsupportedDataType,
            "Linear weight must use Float32 data type"
        ));
    }

    if (!weight.is_contiguous()) {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidWeight, "Linear weight must be contiguous")
        );
    }

    if (bias) {
        if (bias->rank() != 1 || *bias->shape().extent(0) != out_features) {
            return std::unexpected(LayerError(
                LayerErrorCode::InvalidBias,
                "Linear bias must have shape [out_features]"
            ));
        }

        if (bias->data_type() != tensor::DataType::Float32) {
            return std::unexpected(LayerError(
                LayerErrorCode::UnsupportedDataType,
                "Linear bias must use Float32 data type"
            ));
        }

        if (!bias->is_contiguous()) {
            return std::unexpected(
                LayerError(LayerErrorCode::InvalidBias, "Linear bias must be contiguous")
            );
        }
    }

    return Linear(std::move(weight), std::move(bias));
}

std::expected<tensor::Tensor, LayerError> Linear::forward(const tensor::Tensor & input) const
{
    if (input.rank() == 0) {
        return std::unexpected(LayerError(
            LayerErrorCode::InvalidInput,
            "Linear input must have at least one dimension"
        ));
    }

    if (!input.is_contiguous()) {
        return std::unexpected(
            LayerError(LayerErrorCode::InvalidInput, "Linear input must be contiguous")
        );
    }

    if (input.data_type() != tensor::DataType::Float32) {
        return std::unexpected(LayerError(
            LayerErrorCode::UnsupportedDataType,
            "Linear input must use Float32 data type"
        ));
    }

    const auto input_shape = input.shape().values();
    if (input_shape.back() != in_features()) {
        return std::unexpected(LayerError(
            LayerErrorCode::InvalidInput,
            "Linear input feature dimension does not match the weight"
        ));
    }

    std::vector<std::size_t> output_dimensions(input_shape.begin(), input_shape.end());
    output_dimensions.back() = out_features();

    auto output = tensor::Tensor::allocate(
        tensor::DataType::Float32,
        tensor::Shape(std::move(output_dimensions))
    );
    if (!output) {
        return std::unexpected(std::move(output).error());
    }

    auto input_values = input.data_as<float>();
    if (!input_values) {
        return std::unexpected(std::move(input_values).error());
    }

    auto weight_values = weight_.data_as<float>();
    if (!weight_values) {
        return std::unexpected(std::move(weight_values).error());
    }

    auto output_values = output->data_as<float>();
    if (!output_values) {
        return std::unexpected(std::move(output_values).error());
    }

    const std::size_t tokens = input.numel() / in_features();
    if (tokens > 0) {
        kernels::matmul_transposed_rhs_f32(
            *input_values,
            *weight_values,
            *output_values,
            tokens,
            in_features(),
            out_features()
        );

        if (bias_) {
            auto bias_values = bias_->data_as<float>();
            if (!bias_values) {
                return std::unexpected(std::move(bias_values).error());
            }

            for (std::size_t token = 0; token < tokens; ++token) {
                for (std::size_t feature = 0; feature < out_features(); ++feature) {
                    (*output_values)[token * out_features() + feature] += (*bias_values)[feature];
                }
            }
        }
    }

    return std::move(*output);
}

std::size_t Linear::in_features() const noexcept
{
    // 创建全连接层时，权重矩阵的秩已经验证，这里认为一定不会越界
    return *weight_.shape().extent(1);
}

std::size_t Linear::out_features() const noexcept
{
    // 创建全连接层时，权重矩阵的秩已经验证，这里认为一定不会越界
    return *weight_.shape().extent(0);
}

} // namespace liteinfer::core::layer
