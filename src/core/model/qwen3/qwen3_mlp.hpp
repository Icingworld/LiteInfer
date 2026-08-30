#pragma once

#include <cstddef>
#include <expected>

#include "core/layer/linear.hpp"
#include "core/model/model_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::model::qwen3
{

// Qwen3 多层感知机
class Qwen3MLP
{
private:
    explicit Qwen3MLP(layer::Linear gate_proj, layer::Linear up_proj, layer::Linear down_proj);

public:
    // 创建 Qwen3 多层感知机
    [[nodiscard]]
    static std::expected<Qwen3MLP, ModelError>
    create(layer::Linear gate_proj, layer::Linear up_proj, layer::Linear down_proj);

    // 前向传播
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward(const tensor::Tensor & input) const;

    // 获取隐藏层大小
    [[nodiscard]]
    std::size_t hidden_size() const noexcept;

    // 获取中间层大小
    [[nodiscard]]
    std::size_t intermediate_size() const noexcept;

private:
    layer::Linear gate_proj_;
    layer::Linear up_proj_;
    layer::Linear down_proj_;
};

} // namespace liteinfer::core::model::qwen3
