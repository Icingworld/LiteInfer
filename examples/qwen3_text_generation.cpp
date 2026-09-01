#include "core/runtime/qwen3/qwen3_text_generator.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>

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

} // namespace

int main(int argc, char ** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <max_new_tokens> <prompt>\n";
        return 2;
    }

    const std::string_view max_new_tokens_text(argv[2]);
    std::size_t max_new_tokens = 0;
    const auto [end, error] = std::from_chars(
        max_new_tokens_text.data(),
        max_new_tokens_text.data() + max_new_tokens_text.size(),
        max_new_tokens
    );
    if (error != std::errc {} || end != max_new_tokens_text.data() + max_new_tokens_text.size()) {
        std::cerr << "max_new_tokens must be a non-negative integer\n";
        return 2;
    }

    auto filesystem_result =
        filesystem::Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    if (!filesystem_result) {
        print_error("create filesystem", filesystem_result.error());
        return 1;
    }
    auto filesystem = std::move(*filesystem_result);

    const std::filesystem::path model_directory = argv[1];
    auto generator_result = Qwen3TextGenerator::load(filesystem, model_directory);
    if (!generator_result) {
        print_error("load Qwen3 text generator", generator_result.error());
        return 1;
    }
    auto generator = std::move(*generator_result);

    auto output_result = generator.generate(argv[3], max_new_tokens);
    if (!output_result) {
        print_error("generate Qwen3 text", output_result.error());
        return 1;
    }

    std::cout << *output_result << '\n';
    return 0;
}
