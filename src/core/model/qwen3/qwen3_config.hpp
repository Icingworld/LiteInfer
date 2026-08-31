#pragma once
#include <cstddef>
#include <expected>
#include <filesystem>

#include "core/filesystem/filesystem.hpp"
#include "core/json/json.hpp"
#include "core/model/model_error.hpp"

namespace liteinfer::core::model::qwen3
{

// Qwen3 模型架构配置
// Qwen3Config 只负责读取并校验 config.json，不持有模型权重，也不创建网络层
// 当前版本的运行时只支持 Float32、SiLU、default RoPE 和 full attention
class Qwen3Config final
{
private:
    explicit Qwen3Config(
        std::size_t vocab_size,
        std::size_t hidden_size,
        std::size_t intermediate_size,
        std::size_t num_hidden_layers,
        std::size_t num_attention_heads,
        std::size_t num_key_value_heads,
        std::size_t head_dim,
        std::size_t max_position_embeddings,
        float rms_norm_eps,
        float rope_theta,
        bool attention_bias,
        bool tie_word_embeddings
    ) noexcept;

public:
    // 从指定的 config.json 文件加载配置，如果加载失败则返回错误
    [[nodiscard]]
    static std::expected<Qwen3Config, ModelError>
    load(filesystem::Filesystem & filesystem, const std::filesystem::path & path);

    // 从已经解析的 JSON 文档创建配置，如果创建失败则返回错误
    [[nodiscard]]
    static std::expected<Qwen3Config, ModelError> from_json(const json::Document & document);

public:
    // 词汇表大小
    [[nodiscard]]
    std::size_t vocab_size() const noexcept;

    // 隐藏层维度
    [[nodiscard]]
    std::size_t hidden_size() const noexcept;

    // MLP 中间层维度
    [[nodiscard]]
    std::size_t intermediate_size() const noexcept;

    // Transformer DecoderLayer 数量
    [[nodiscard]]
    std::size_t num_hidden_layers() const noexcept;

    // Query attention head 数量
    [[nodiscard]]
    std::size_t num_attention_heads() const noexcept;

    // Key/Value attention head 数量
    [[nodiscard]]
    std::size_t num_key_value_heads() const noexcept;

    // 单个 attention head 的维度
    [[nodiscard]]
    std::size_t head_dim() const noexcept;

    // 支持的最大序列长度
    [[nodiscard]]
    std::size_t max_position_embeddings() const noexcept;

    // RMSNorm epsilon
    [[nodiscard]]
    float rms_norm_eps() const noexcept;

    // RoPE base theta
    [[nodiscard]]
    float rope_theta() const noexcept;

    // Attention projection 是否包含 bias
    [[nodiscard]]
    bool attention_bias() const noexcept;

    // 输入 embedding 和输出 lm_head 是否共享权重
    [[nodiscard]]
    bool tie_word_embeddings() const noexcept;

private:
    std::size_t vocab_size_;
    std::size_t hidden_size_;
    std::size_t intermediate_size_;
    std::size_t num_hidden_layers_;
    std::size_t num_attention_heads_;
    std::size_t num_key_value_heads_;
    std::size_t head_dim_;
    std::size_t max_position_embeddings_;
    float rms_norm_eps_;
    float rope_theta_;
    bool attention_bias_;
    bool tie_word_embeddings_;
};

} // namespace liteinfer::core::model::qwen3
