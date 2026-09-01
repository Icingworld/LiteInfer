#include "core/tokenizer/qwen3/qwen3_tokenizer.hpp"
#include "core/filesystem/filesystem.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif

namespace
{

using namespace liteinfer::core;
using tokenizer::qwen3::Qwen3Tokenizer;

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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <text>\n";
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
    auto tokenizer_result = Qwen3Tokenizer::load(filesystem, model_directory);
    if (!tokenizer_result) {
        print_error("load Qwen3 tokenizer", tokenizer_result.error());
        return 1;
    }
    auto tokenizer = std::move(*tokenizer_result);

    const auto encoded_result = tokenizer.encode(argv[2], false);
    if (!encoded_result) {
        print_error("encode text", encoded_result.error());
        return 1;
    }

    const auto decoded_result = tokenizer.decode(*encoded_result, true);
    if (!decoded_result) {
        print_error("decode token IDs", decoded_result.error());
        return 1;
    }

    std::cout << "ids:";
    for (const auto token_id : *encoded_result) {
        std::cout << ' ' << token_id;
    }
    std::cout << '\n';
    std::cout << "decoded: " << *decoded_result << '\n';
    std::cout << "vocab_size: " << tokenizer.vocab_size() << '\n';
    std::cout << "total_vocab_size: " << tokenizer.total_vocab_size() << '\n';

    return 0;
}
