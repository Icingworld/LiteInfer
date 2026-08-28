#pragma once

#include <cstddef>
#include <expected>
#include <optional>

#include "core/layer/layer_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::layer
{

// 全连接层
// 当前版本只支持 float32 类型的输入和输出
class Linear
{
private:
    explicit Linear(tensor::Tensor weight, std::optional<tensor::Tensor> bias);

public:
    // 创建全连接层
    // weight: 权重 [out_features, in_features]
    // bias: 偏置 [out_features] 可选
    [[nodiscard]]
    static std::expected<Linear, LayerError>
    create(tensor::Tensor weight, std::optional<tensor::Tensor> bias = std::nullopt);

    // 前向传播
    // input: 输入 [..., in_features] 最后一维为输入特征维
    // output: 输出 [..., out_features] 最后一维为输出特征维，计算失败则返回错误
    [[nodiscard]]
    std::expected<tensor::Tensor, LayerError> forward(const tensor::Tensor & input) const;

    // 输入特征数量
    [[nodiscard]]
    std::size_t in_features() const noexcept;

    // 输出特征数量
    [[nodiscard]]
    std::size_t out_features() const noexcept;

private:
    tensor::Tensor weight_;
    std::optional<tensor::Tensor> bias_;
};

} // namespace liteinfer::core::layer
