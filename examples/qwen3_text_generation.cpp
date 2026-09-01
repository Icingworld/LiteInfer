#include "core/runtime/qwen3/qwen3_text_generator.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <limits>
#include <windows.h>
#endif

#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif

namespace
{

using namespace liteinfer::core;
using runtime::qwen3::Qwen3TextGenerator;

#if defined(_WIN32)
using NativeFilesystemBackend = filesystem::backend::windows::WindowsFilesystemBackend;
#else
using NativeFilesystemBackend = filesystem::backend::posix::PosixFilesystemBackend;
#endif

void print_error(std::string_view operation, const common::Error & error)
{
    std::cerr << operation << " failed: " << error.message() << '\n';
}

void print_usage()
{
    std::cerr << "Usage: liteinfer_qwen3_text_generation"
              << " <model_dir> <max_new_tokens> <prompt>\n";
}

std::optional<std::size_t> parse_max_new_tokens(std::string_view text)
{
    std::size_t max_new_tokens = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), max_new_tokens);
    if (error != std::errc {} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return max_new_tokens;
}

int generate_text(
    const std::filesystem::path & model_directory,
    std::size_t max_new_tokens,
    std::string_view prompt
)
{
    auto filesystem_result =
        filesystem::Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    if (!filesystem_result) {
        print_error("create filesystem", filesystem_result.error());
        return 1;
    }
    auto filesystem = std::move(*filesystem_result);

    auto generator_result = Qwen3TextGenerator::load(filesystem, model_directory);
    if (!generator_result) {
        print_error("load Qwen3 text generator", generator_result.error());
        return 1;
    }
    auto generator = std::move(*generator_result);

    auto output_result = generator.generate(prompt, max_new_tokens);
    if (!output_result) {
        print_error("generate Qwen3 text", output_result.error());
        return 1;
    }

    std::cout << *output_result << '\n';
    return 0;
}

#if defined(_WIN32)
std::optional<std::string> wide_to_utf8(std::wstring_view text)
{
    if (text.empty()) {
        return std::string {};
    }
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }

    const auto input_length = static_cast<int>(text.size());
    const int output_length = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        text.data(),
        input_length,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    if (output_length <= 0) {
        return std::nullopt;
    }

    std::string result(static_cast<std::size_t>(output_length), '\0');
    const int converted_length = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        text.data(),
        input_length,
        result.data(),
        output_length,
        nullptr,
        nullptr
    );
    if (converted_length != output_length) {
        return std::nullopt;
    }
    return result;
}
#endif

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t ** argv)
{
    if (argc != 4) {
        print_usage();
        return 2;
    }

    const auto max_new_tokens_text = wide_to_utf8(argv[2]);
    const auto prompt = wide_to_utf8(argv[3]);
    if (!max_new_tokens_text || !prompt) {
        std::cerr << "Command-line arguments must be valid UTF-16\n";
        return 2;
    }

    const auto max_new_tokens = parse_max_new_tokens(*max_new_tokens_text);
    if (!max_new_tokens) {
        std::cerr << "max_new_tokens must be a non-negative integer\n";
        return 2;
    }

    return generate_text(std::filesystem::path(argv[1]), *max_new_tokens, *prompt);
}
#else
int main(int argc, char ** argv)
{
    if (argc != 4) {
        print_usage();
        return 2;
    }

    const auto max_new_tokens = parse_max_new_tokens(argv[2]);
    if (!max_new_tokens) {
        std::cerr << "max_new_tokens must be a non-negative integer\n";
        return 2;
    }

    return generate_text(std::filesystem::path(argv[1]), *max_new_tokens, argv[3]);
}
#endif
