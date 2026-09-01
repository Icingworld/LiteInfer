#pragma once

#include <cstddef>
#include <expected>

#include "core/kvcache/kvcache.hpp"
#include "core/layer/rms_norm.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_attention.hpp"
#include "core/model/qwen3/qwen3_mlp.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::model::qwen3
{

// Qwen3 解码器层
// 使用 Pre-Norm 结构依次执行自注意力和 MLP，并在两个子层后添加残差连接
class Qwen3DecoderLayer
{
private:
    explicit Qwen3DecoderLayer(
        Qwen3Attention self_attn,
        Qwen3MLP mlp,
        layer::RMSNorm input_layernorm,
        layer::RMSNorm post_attention_layernorm
    );

public:
    // 创建 Qwen3 DecoderLayer
    [[nodiscard]]
    static std::expected<Qwen3DecoderLayer, ModelError> create(
        Qwen3Attention self_attn,
        Qwen3MLP mlp,
        layer::RMSNorm input_layernorm,
        layer::RMSNorm post_attention_layernorm
    );

    // 前向传播
    // input: [..., sequence_length, hidden_size]
    // output: 与 input 形状相同
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward(const tensor::Tensor & input) const;

    // 使用指定 cache region 执行当前 DecoderLayer。
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward(
        const tensor::Tensor & input,
        kvcache::KVCache & cache,
        std::size_t layer_index,
        const kvcache::KVCacheRegion & region
    ) const;

    // 获取隐藏层大小
    [[nodiscard]]
    std::size_t hidden_size() const noexcept;

    // 获取 MLP 中间层大小
    [[nodiscard]]
    std::size_t intermediate_size() const noexcept;

    // 获取块级 RMSNorm 的 epsilon
    [[nodiscard]]
    float rms_norm_eps() const noexcept;

private:
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward_impl(
        const tensor::Tensor & input,
        kvcache::KVCache * cache,
        std::size_t layer_index,
        const kvcache::KVCacheRegion * region
    ) const;

    Qwen3Attention self_attn_;
    Qwen3MLP mlp_;
    layer::RMSNorm input_layernorm_;
    layer::RMSNorm post_attention_layernorm_;
};

} // namespace liteinfer::core::model::qwen3
