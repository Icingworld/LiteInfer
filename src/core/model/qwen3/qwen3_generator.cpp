#include "core/model/qwen3/qwen3_generator.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace liteinfer::core::model::qwen3
{

namespace
{

ModelError invalid_input(std::string_view message)
{
    return ModelError(ModelErrorCode::InvalidInput, message);
}

ModelError invalid_configuration(std::string_view message)
{
    return ModelError(ModelErrorCode::InvalidConfiguration, message);
}

std::expected<tensor::Tensor, ModelError> make_input_tensor(
    const std::vector<common::TokenId> & token_ids
)
{
    auto input = tensor::Tensor::allocate(
        common::data_type::DataType::Int64,
        tensor::Shape {1, token_ids.size()}
    );
    if (!input) [[unlikely]] {
        return std::unexpected(std::move(input).error());
    }

    auto values = input->data_as<std::int64_t>();
    if (!values) [[unlikely]] {
        return std::unexpected(std::move(values).error());
    }

    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        (*values)[index] = static_cast<std::int64_t>(token_ids[index]);
    }

    return std::move(*input);
}

std::expected<common::TokenId, ModelError>
select_greedy_token(const tensor::Tensor & logits, std::size_t vocab_size)
{
    if (vocab_size == 0) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model returned an empty logits dimension")
        );
    }

    const auto shape = logits.shape().values();
    const bool expected_shape =
        (shape.size() == 2 && shape[0] == 1 && shape[1] == vocab_size) ||
        (shape.size() == 3 && shape[0] == 1 && shape[1] == 1 && shape[2] == vocab_size);
    if (!expected_shape) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model returned an unexpected logits shape")
        );
    }

    if (logits.data_type() != common::data_type::DataType::Float32) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model logits must use Float32 data type")
        );
    }

    if (logits.numel() != vocab_size) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model returned inconsistent logits storage")
        );
    }

    auto values = logits.data_as<float>();
    if (!values) [[unlikely]] {
        return std::unexpected(std::move(values).error());
    }

    const auto last_logits = values->subspan(0, vocab_size);

    for (const float score : last_logits) {
        if (!std::isfinite(score)) [[unlikely]] {
            return std::unexpected(invalid_configuration("Qwen3Model returned a non-finite logit"));
        }
    }

    common::TokenId best_token = 0;
    float best_score = last_logits[0];

    // 只在严格变大时更新，因此相同分数时保留较小的 token ID。
    for (std::size_t token_id = 1; token_id < vocab_size; ++token_id) {
        if (last_logits[token_id] > best_score) {
            best_score = last_logits[token_id];
            best_token = static_cast<common::TokenId>(token_id);
        }
    }

    return best_token;
}

} // namespace

Qwen3Generator::Qwen3Generator(const Qwen3Model & model) noexcept
    : model_(&model)
{}

std::expected<std::vector<common::TokenId>, ModelError> Qwen3Generator::generate(
    std::span<const common::TokenId> prompt,
    std::size_t max_new_tokens,
    std::optional<common::TokenId> eos_token_id
) const
{
    if (prompt.empty()) [[unlikely]] {
        return std::unexpected(invalid_input("Qwen3Generator prompt must not be empty"));
    }

    const std::size_t vocab_size = model_->vocab_size();
    if (vocab_size == 0) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model vocabulary size must not be zero")
        );
    }

    // TokenId 是 uint32_t，确保 logits 的索引能够安全转换回 TokenId。
    if (vocab_size - 1 > std::numeric_limits<common::TokenId>::max()) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3Model vocabulary is too large for TokenId")
        );
    }

    if (prompt.size() > model_->config().max_position_embeddings()) [[unlikely]] {
        return std::unexpected(
            invalid_input("Qwen3Generator prompt exceeds max_position_embeddings")
        );
    }

    if (max_new_tokens > model_->config().max_position_embeddings() - prompt.size()) [[unlikely]] {
        return std::unexpected(
            invalid_input("Qwen3Generator requested sequence exceeds max_position_embeddings")
        );
    }

    if (eos_token_id.has_value() && static_cast<std::size_t>(*eos_token_id) >= vocab_size)
        [[unlikely]] {
        return std::unexpected(invalid_input("Qwen3Generator EOS token is out of vocabulary"));
    }

    std::vector<common::TokenId> tokens(prompt.begin(), prompt.end());
    tokens.reserve(tokens.size() + max_new_tokens);

    if (max_new_tokens == 0) {
        return tokens;
    }

    auto cache = model_->create_kv_cache();
    if (!cache) [[unlikely]] {
        return std::unexpected(std::move(cache).error());
    }

    auto prompt_input = make_input_tensor(tokens);
    if (!prompt_input) [[unlikely]] {
        return std::unexpected(std::move(prompt_input).error());
    }

    auto logits = model_->prefill(*prompt_input, *cache);
    if (!logits) [[unlikely]] {
        return std::unexpected(std::move(logits).error());
    }

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto next_token = select_greedy_token(*logits, vocab_size);
        if (!next_token) [[unlikely]] {
            return std::unexpected(std::move(next_token).error());
        }

        tokens.push_back(*next_token);
        if (eos_token_id.has_value() && *next_token == *eos_token_id) {
            break;
        }

        if (step + 1 == max_new_tokens) {
            break;
        }

        auto decode_input = make_input_tensor(std::vector<common::TokenId> {*next_token});
        if (!decode_input) [[unlikely]] {
            return std::unexpected(std::move(decode_input).error());
        }

        logits = model_->decode(*decode_input, *cache);
        if (!logits) [[unlikely]] {
            return std::unexpected(std::move(logits).error());
        }
    }

    return tokens;
}

} // namespace liteinfer::core::model::qwen3
