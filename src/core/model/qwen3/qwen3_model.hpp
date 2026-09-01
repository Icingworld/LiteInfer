#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <vector>

#include "core/embedding/embedding.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/kvcache/kvcache.hpp"
#include "core/layer/linear.hpp"
#include "core/layer/rms_norm.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_config.hpp"
#include "core/model/qwen3/qwen3_decoder_layer.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::model::qwen3
{

// Qwen3 基础语言模型
// 当前版本支持 Float32、full attention、完整序列推理以及连续静态 KV cache 推理
class Qwen3Model final
{
private:
    explicit Qwen3Model(
        Qwen3Config config,
        embedding::Embedding embed_tokens,
        std::vector<Qwen3DecoderLayer> layers,
        layer::RMSNorm norm,
        layer::Linear lm_head
    );

public:
    // 从模型目录加载 config.json 和 model.safetensors，并组装完整网络
    [[nodiscard]]
    static std::expected<Qwen3Model, ModelError>
    load(filesystem::Filesystem & filesystem, const std::filesystem::path & model_directory);

    // 执行一次完整序列前向传播
    // token_ids: [sequence_length] 或 [batch_size, sequence_length]
    // output: [sequence_length, vocab_size] 或 [batch_size, sequence_length, vocab_size]
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward(const tensor::Tensor & token_ids) const;

    // 为单请求生成一个与模型配置匹配的静态 KV cache。
    [[nodiscard]]
    std::expected<kvcache::KVCache, ModelError> create_kv_cache() const;

    // 处理一段输入并将每层产生的 K/V 写入 cache，返回最后一个位置的 logits [1, vocab_size]
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError>
    prefill(const tensor::Tensor & token_ids, kvcache::KVCache & cache) const;

    // 处理一个新 token 并追加到 cache，返回当前 token 的 logits [1, vocab_size]
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError>
    decode(const tensor::Tensor & token_ids, kvcache::KVCache & cache) const;

    // 获取模型配置
    [[nodiscard]]
    const Qwen3Config & config() const noexcept;

    // 获取词汇表大小
    [[nodiscard]]
    std::size_t vocab_size() const noexcept;

    // 获取隐藏层大小
    [[nodiscard]]
    std::size_t hidden_size() const noexcept;

    // 获取 DecoderLayer 数量
    [[nodiscard]]
    std::size_t num_hidden_layers() const noexcept;

private:
    [[nodiscard]]
    std::expected<tensor::Tensor, ModelError> forward_cached(
        const tensor::Tensor & token_ids,
        kvcache::KVCache & cache,
        bool decode_mode
    ) const;

    Qwen3Config config_;
    embedding::Embedding embed_tokens_;
    std::vector<Qwen3DecoderLayer> layers_;
    layer::RMSNorm norm_;
    layer::Linear lm_head_;
};

} // namespace liteinfer::core::model::qwen3
