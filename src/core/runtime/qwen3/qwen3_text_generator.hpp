#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

#include "core/common/error.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/model/qwen3/qwen3_model.hpp"
#include "core/tokenizer/qwen3/qwen3_tokenizer.hpp"

namespace liteinfer::core::runtime::qwen3
{

// Qwen3 单次文本生成 facade
// prompt 被视为已经准备好的完整输入，不在此处维护历史或套用 chat template
class Qwen3TextGenerator final
{
public:
    // 从同一个模型目录加载 tokenizer 和模型权重
    [[nodiscard]]
    static std::expected<Qwen3TextGenerator, common::Error>
    load(filesystem::Filesystem & filesystem, const std::filesystem::path & model_directory);

    Qwen3TextGenerator(const Qwen3TextGenerator &) = delete;

    Qwen3TextGenerator & operator=(const Qwen3TextGenerator &) = delete;

    Qwen3TextGenerator(Qwen3TextGenerator &&) noexcept = default;

    Qwen3TextGenerator & operator=(Qwen3TextGenerator &&) noexcept = default;

    ~Qwen3TextGenerator() = default;

public:
    // 返回新生成的文本，不包含 prompt 本身
    // max_new_tokens 为 0 时返回空字符串
    [[nodiscard]]
    std::expected<std::string, common::Error>
    generate(std::string_view prompt, std::size_t max_new_tokens) const;

private:
    Qwen3TextGenerator(
        tokenizer::qwen3::Qwen3Tokenizer tokenizer,
        model::qwen3::Qwen3Model model
    ) noexcept;

private:
    tokenizer::qwen3::Qwen3Tokenizer tokenizer_;
    model::qwen3::Qwen3Model model_;
};

} // namespace liteinfer::core::runtime::qwen3
