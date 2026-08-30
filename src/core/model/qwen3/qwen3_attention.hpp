#pragma once

#include <cstddef>
#include <expected>

#include "core/layer/linear.hpp"
#include "core/layer/rms_norm.hpp"
#include "core/model/model_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::model::qwen3
{

// Qwen3 分组查询自注意力
// 当前版本只支持不带 KVCache 的 Float32 causal attention
class Qwen3Attention
{
private:
    explicit Qwen3Attention(
        layer::Linear q_proj,
        layer::Linear k_proj,
        layer::Linear v_proj,
        layer::Linear o_proj,
        layer::RMSNorm q_norm,
        layer::RMSNorm k_norm,
        std::size_t num_attention_heads,
        std::size_t num_key_value_heads,
        std::size_t head_dim,
        float rope_theta
    );

public:
    // 创建 Qwen3 Attention
    [[nodiscard]]
    static std::expected<Qwen3Attention, ModelError> create(
        layer::Linear q_proj,
        layer::Linear k_proj,
        layer::Linear v_proj,
        layer::Linear o_proj,
        layer::RMSNorm q_norm,
        layer::RMSNorm k_norm,
        std::size_t num_attention_heads,
        std::size_t num_key_value_heads,
        float rope_theta
    );

    // 前向传播
    // input: [..., sequence_length, hidden_size]
    // output: 与 input 形状相同
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward(const tensor::Tensor & input) const;

    // 获取隐藏层大小
    [[nodiscard]]
    std::size_t hidden_size() const noexcept;

    // 获取注意力头数
    [[nodiscard]]
    std::size_t num_attention_heads() const noexcept;

    // 获取键值对头数
    [[nodiscard]]
    std::size_t num_key_value_heads() const noexcept;

    // 获取头维度
    [[nodiscard]]
    std::size_t head_dim() const noexcept;

    // 获取 RoPE 参数
    [[nodiscard]]
    float rope_theta() const noexcept;

private:
    layer::Linear q_proj_;
    layer::Linear k_proj_;
    layer::Linear v_proj_;
    layer::Linear o_proj_;
    layer::RMSNorm q_norm_;
    layer::RMSNorm k_norm_;
    std::size_t num_attention_heads_;
    std::size_t num_key_value_heads_;
    std::size_t head_dim_;
    float rope_theta_;
};

} // namespace liteinfer::core::model::qwen3
