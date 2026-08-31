#pragma once

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "core/common/types.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_model.hpp"

namespace liteinfer::core::model::qwen3
{

// 基于 Qwen3Model 的 token ID 级 greedy generation
// 当前版本每一步都会重新计算完整序列，不使用 KV cache
class Qwen3Generator final
{
public:
    // model 必须在当前 Generator 的整个生命周期内保持有效
    explicit Qwen3Generator(const Qwen3Model & model) noexcept;

public:
    // 返回值包含原始 prompt 以及新生成的 token
    // 当生成出的 token 等于 eos_token_id 时提前停止
    [[nodiscard]]
    std::expected<std::vector<common::TokenId>, ModelError> generate(
        std::span<const common::TokenId> prompt,
        std::size_t max_new_tokens,
        std::optional<common::TokenId> eos_token_id = std::nullopt
    ) const;

private:
    const Qwen3Model * model_;
};

} // namespace liteinfer::core::model::qwen3
