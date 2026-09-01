#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/filesystem/filesystem.hpp"
#include "core/tokenizer/tokenizer.hpp"

namespace liteinfer::core::tokenizer::qwen3
{

// Qwen3 的 Qwen2Tokenizer 兼容 Byte-Level BPE 实现
class Qwen3Tokenizer final : public Tokenizer
{
public:
    // 从模型目录加载 tokenizer.json 和 tokenizer_config.json
    [[nodiscard]]
    static std::expected<Qwen3Tokenizer, TokenizerError>
    load(filesystem::Filesystem & filesystem, const std::filesystem::path & model_directory);

    Qwen3Tokenizer(const Qwen3Tokenizer &) = delete;

    Qwen3Tokenizer & operator=(const Qwen3Tokenizer &) = delete;

    Qwen3Tokenizer(Qwen3Tokenizer &&) noexcept;

    Qwen3Tokenizer & operator=(Qwen3Tokenizer &&) noexcept;

    ~Qwen3Tokenizer() override;

public:
    [[nodiscard]]
    std::expected<std::vector<common::TokenId>, TokenizerError>
    encode(std::string_view text, bool add_special_tokens = false) const override;

    [[nodiscard]]
    std::expected<std::string, TokenizerError> decode(
        std::span<const common::TokenId> token_ids,
        bool skip_special_tokens = true
    ) const override;

    [[nodiscard]]
    std::size_t vocab_size() const noexcept override;

    // 获取基础词表和 added tokens 合并后的最大已知 ID 范围
    [[nodiscard]]
    std::size_t total_vocab_size() const noexcept;

    [[nodiscard]]
    std::optional<common::TokenId> token_id(std::string_view token) const noexcept override;

    [[nodiscard]]
    std::optional<common::TokenId> bos_token_id() const noexcept;

    [[nodiscard]]
    std::optional<common::TokenId> eos_token_id() const noexcept;

    [[nodiscard]]
    std::optional<common::TokenId> pad_token_id() const noexcept;

    struct Impl;

private:
    explicit Qwen3Tokenizer(std::unique_ptr<Impl> impl) noexcept;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace liteinfer::core::tokenizer::qwen3
