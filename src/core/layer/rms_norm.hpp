#pragma once

#include <cstddef>
#include <expected>

#include "core/layer/layer_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::layer
{

// RMS Norm 层
// 当前版本只支持 float32 类型的输入和输出，并沿最后一维进行归一化
class RMSNorm
{
private:
    explicit RMSNorm(tensor::Tensor weight, float eps);

public:
    // 创建 RMS Norm
    // weight: 缩放权重 [normalized_size]
    // eps: 有限且大于 0 的极小值，防止除以 0
    [[nodiscard]]
    static std::expected<RMSNorm, LayerError> create(tensor::Tensor weight, float eps);

    // 前向传播
    // input: 输入 [..., normalized_size]
    // output: 与输入形状相同的归一化结果，计算失败则返回错误
    [[nodiscard]]
    std::expected<tensor::Tensor, LayerError> forward(const tensor::Tensor & input) const;

    // 归一化维度
    [[nodiscard]]
    std::size_t normalized_size() const noexcept;

    // 极小值，防止除以 0
    [[nodiscard]]
    float eps() const noexcept;

private:
    tensor::Tensor weight_;
    float eps_;
};

} // namespace liteinfer::core::layer
