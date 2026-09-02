#include "core/model/qwen3/qwen3_model.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "core/filesystem/filesystem.hpp"

#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif

namespace
{

using namespace liteinfer::core;
using model::qwen3::Qwen3Model;

#if defined(_WIN32)
using NativeFilesystemBackend = filesystem::backend::windows::WindowsFilesystemBackend;
#else
using NativeFilesystemBackend = filesystem::backend::posix::PosixFilesystemBackend;
#endif

void print_error(std::string_view operation, const common::Error & error)
{
    std::cerr << operation << " failed: " << error.message() << '\n';
}

bool parse_token_id(std::string_view text, std::int64_t & value)
{
    if (text.empty()) {
        return false;
    }

    const auto * begin = text.data();
    const auto * end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc {} && result.ptr == end && value >= 0;
}

void print_shape(const tensor::Tensor & value)
{
    const auto dimensions = value.shape().values();
    std::cout << "output shape: [";
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << dimensions[index];
    }
    std::cout << "]\n";
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <output.f32> <token_id> [token_id ...]\n";
        return 2;
    }

    const std::filesystem::path model_directory = argv[1];
    const std::filesystem::path output_path = argv[2];

    std::vector<std::int64_t> token_values;
    token_values.reserve(static_cast<std::size_t>(argc - 3));
    for (int index = 3; index < argc; ++index) {
        std::int64_t token_id = 0;
        if (!parse_token_id(argv[index], token_id)) {
            std::cerr << "Invalid token ID: " << argv[index] << '\n';
            return 2;
        }
        token_values.push_back(token_id);
    }

    auto filesystem_result =
        filesystem::Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    if (!filesystem_result) {
        print_error("create filesystem", filesystem_result.error());
        return 1;
    }
    auto filesystem = std::move(*filesystem_result);

    auto model_result = Qwen3Model::load(filesystem, model_directory);
    if (!model_result) {
        print_error("load Qwen3 model", model_result.error());
        return 1;
    }
    auto model = std::move(*model_result);

    auto token_ids_result = tensor::Tensor::allocate(
        common::data_type::DataType::Int64,
        tensor::Shape {1, token_values.size()}
    );
    if (!token_ids_result) {
        print_error("allocate token IDs", token_ids_result.error());
        return 1;
    }
    auto token_ids = std::move(*token_ids_result);

    auto token_ids_data = token_ids.data_as<std::int64_t>();
    if (!token_ids_data) {
        print_error("access token IDs", token_ids_data.error());
        return 1;
    }
    std::copy(token_values.begin(), token_values.end(), token_ids_data->begin());

    auto logits_result = model.forward(token_ids);
    if (!logits_result) {
        print_error("run Qwen3 forward", logits_result.error());
        return 1;
    }
    auto logits = std::move(*logits_result);

    auto logits_data = logits.data_as<float>();
    if (!logits_data) {
        print_error("access logits", logits_data.error());
        return 1;
    }

    if (logits.empty()) {
        std::cerr << "Qwen3 forward returned empty logits\n";
        return 1;
    }

    std::error_code directory_error;
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path(), directory_error);
        if (directory_error) {
            std::cerr << "create output directory failed: " << directory_error.message() << '\n';
            return 1;
        }
    }

    std::ofstream output_stream(output_path, std::ios::binary | std::ios::trunc);
    if (!output_stream.is_open()) {
        std::cerr << "open logits output failed: " << output_path << '\n';
        return 1;
    }
    output_stream.write(
        reinterpret_cast<const char *>(logits_data->data()),
        static_cast<std::streamsize>(logits_data->size_bytes())
    );
    if (!output_stream.good()) {
        std::cerr << "write logits output failed: " << output_path << '\n';
        return 1;
    }

    const auto shape = logits.shape().values();
    const auto shape_path = std::filesystem::path(output_path.string() + ".shape");
    std::ofstream shape_stream(shape_path, std::ios::trunc);
    if (!shape_stream.is_open()) {
        std::cerr << "open logits shape output failed: " << shape_path << '\n';
        return 1;
    }
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            shape_stream << ' ';
        }
        shape_stream << shape[index];
    }
    shape_stream << '\n';
    if (!shape_stream.good()) {
        std::cerr << "write logits shape output failed: " << shape_path << '\n';
        return 1;
    }

    print_shape(logits);
    std::cout << "wrote logits: " << output_path << '\n';
    std::cout << "wrote shape: " << shape_path << '\n';

    const std::size_t vocab_size = model.vocab_size();
    const auto last_position = logits_data->subspan(logits_data->size() - vocab_size);
    const auto best = std::max_element(last_position.begin(), last_position.end());
    const auto next_token = static_cast<std::size_t>(std::distance(last_position.begin(), best));
    std::cout << "greedy next token ID: " << next_token << '\n';

    return 0;
}
