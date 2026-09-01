#include "core/runtime/qwen3/qwen3_text_generator.hpp"

#include <span>
#include <string>
#include <utility>

#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_generator.hpp"

namespace liteinfer::core::runtime::qwen3
{

namespace
{

common::Error invalid_input(std::string_view message)
{
    return common::Error(model::ModelErrorCode::InvalidInput, message);
}

common::Error invalid_configuration(std::string_view message)
{
    return common::Error(model::ModelErrorCode::InvalidConfiguration, message);
}

} // namespace

Qwen3TextGenerator::Qwen3TextGenerator(
    tokenizer::qwen3::Qwen3Tokenizer tokenizer,
    model::qwen3::Qwen3Model model
) noexcept
    : tokenizer_(std::move(tokenizer))
    , model_(std::move(model))
{}

std::expected<Qwen3TextGenerator, common::Error> Qwen3TextGenerator::load(
    filesystem::Filesystem & filesystem,
    const std::filesystem::path & model_directory
)
{
    auto model_result = model::qwen3::Qwen3Model::load(filesystem, model_directory);
    if (!model_result) [[unlikely]] {
        return std::unexpected(std::move(model_result).error());
    }

    auto tokenizer_result = tokenizer::qwen3::Qwen3Tokenizer::load(filesystem, model_directory);
    if (!tokenizer_result) [[unlikely]] {
        return std::unexpected(std::move(tokenizer_result).error());
    }

    auto model = std::move(*model_result);
    auto tokenizer = std::move(*tokenizer_result);
    if (tokenizer.total_vocab_size() > model.vocab_size()) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3 tokenizer vocabulary is larger than the model vocabulary")
        );
    }

    return Qwen3TextGenerator(std::move(tokenizer), std::move(model));
}

std::expected<std::string, common::Error>
Qwen3TextGenerator::generate(std::string_view prompt, std::size_t max_new_tokens) const
{
    if (prompt.empty()) [[unlikely]] {
        return std::unexpected(invalid_input("Qwen3 text generation prompt must not be empty"));
    }

    auto prompt_ids_result = tokenizer_.encode(prompt, false);
    if (!prompt_ids_result) [[unlikely]] {
        return std::unexpected(std::move(prompt_ids_result).error());
    }
    if (prompt_ids_result->empty()) [[unlikely]] {
        return std::unexpected(
            invalid_input("Qwen3 tokenizer produced an empty prompt token sequence")
        );
    }

    for (const auto token_id : *prompt_ids_result) {
        if (static_cast<std::size_t>(token_id) >= model_.vocab_size()) [[unlikely]] {
            return std::unexpected(
                invalid_input("Qwen3 tokenizer produced a token ID outside the model vocabulary")
            );
        }
    }

    // Qwen3Generator 只借用 model_，在本次调用内构造可以避免移动 facade 后留下悬空指针
    model::qwen3::Qwen3Generator generator(model_);
    auto generated_ids_result =
        generator.generate(*prompt_ids_result, max_new_tokens, tokenizer_.eos_token_id());
    if (!generated_ids_result) [[unlikely]] {
        return std::unexpected(std::move(generated_ids_result).error());
    }

    const auto generated_ids =
        std::span<const common::TokenId>(*generated_ids_result).subspan(prompt_ids_result->size());
    auto decoded_result = tokenizer_.decode(generated_ids, true);
    if (!decoded_result) [[unlikely]] {
        return std::unexpected(std::move(decoded_result).error());
    }

    return std::move(*decoded_result);
}

} // namespace liteinfer::core::runtime::qwen3
