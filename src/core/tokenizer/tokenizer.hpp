#pragma once

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/types.hpp"
#include "core/tokenizer/tokenizer_error.hpp"

namespace liteinfer::core::tokenizer
{

// Tokenizer 的通用接口
// 具体模型可以根据自己的词表格式提供实现
class Tokenizer
{
public:
    virtual ~Tokenizer() = default;

public:
    // 将 UTF-8 文本编码为 token ID
    [[nodiscard]]
    virtual std::expected<std::vector<common::TokenId>, TokenizerError>
    encode(std::string_view text, bool add_special_tokens = false) const = 0;

    // 将 token ID 解码为 UTF-8 文本
    [[nodiscard]]
    virtual std::expected<std::string, TokenizerError>
    decode(std::span<const common::TokenId> token_ids, bool skip_special_tokens = true) const = 0;

    // 获取基础 BPE 词表大小，不包含 added tokens
    [[nodiscard]]
    virtual std::size_t vocab_size() const noexcept = 0;

    // 查找一个 token 的 ID
    [[nodiscard]]
    virtual std::optional<common::TokenId> token_id(std::string_view token) const noexcept = 0;
};

} // namespace liteinfer::core::tokenizer
