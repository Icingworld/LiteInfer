#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string_view>
#include <utility>

#include "core/common/error.hpp"
#include "core/filesystem/backend/filesystem_backend.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/filesystem/filesystem_error.hpp"
#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif
#include "core/json/json.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_config.hpp"

namespace
{

using namespace liteinfer::core;
using filesystem::Filesystem;
using model::ModelErrorCode;
using model::qwen3::Qwen3Config;

#if defined(_WIN32)
using NativeFilesystemBackend = filesystem::backend::windows::WindowsFilesystemBackend;
#else
using NativeFilesystemBackend = filesystem::backend::posix::PosixFilesystemBackend;
#endif

json::Document make_valid_document()
{
    return json::Document::parse(
        R"({
            "model_type": "qwen3",
            "attention_bias": false,
            "dtype": "float32",
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 128,
            "intermediate_size": 384,
            "layer_types": ["full_attention", "full_attention"],
            "max_position_embeddings": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "rms_norm_eps": 1e-6,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "vocab_size": 1024
        })"
    );
}

Filesystem make_filesystem()
{
    auto result = Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    assert(result.has_value());
    return std::move(*result);
}

class TemporaryFixture
{
public:
    TemporaryFixture()
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_qwen3_config_test_" +
#if defined(_WIN32)
               std::to_string(static_cast<unsigned long long>(::_getpid()))
#else
               std::to_string(static_cast<unsigned long long>(::getpid()))
#endif
              )
          )
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        assert(!error);
        assert(std::filesystem::create_directories(directory_));
    }

    TemporaryFixture(const TemporaryFixture &) = delete;
    TemporaryFixture & operator=(const TemporaryFixture &) = delete;

    ~TemporaryFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    [[nodiscard]]
    std::filesystem::path write(std::string_view content) const
    {
        const auto path = directory_ / "config.json";
        std::ofstream stream(path);
        assert(stream.is_open());
        stream << content;
        assert(stream.good());
        return path;
    }

    [[nodiscard]]
    const std::filesystem::path & directory() const noexcept
    {
        return directory_;
    }

private:
    std::filesystem::path directory_;
};

void assert_invalid_config(const json::Document & document)
{
    auto result = Qwen3Config::from_json(document);
    assert(!result.has_value());
    assert(result.error().category() == common::ErrorCategory::Model);
    assert(result.error().code() == std::to_underlying(ModelErrorCode::InvalidConfiguration));
}

void assert_tiny_config(const Qwen3Config & config)
{
    assert(config.vocab_size() == 1024);
    assert(config.hidden_size() == 128);
    assert(config.intermediate_size() == 384);
    assert(config.num_hidden_layers() == 2);
    assert(config.num_attention_heads() == 4);
    assert(config.num_key_value_heads() == 2);
    assert(config.head_dim() == 32);
    assert(config.max_position_embeddings() == 256);
    assert(config.rms_norm_eps() == 1.0e-6F);
    assert(config.rope_theta() == 1'000'000.0F);
    assert(!config.attention_bias());
    assert(config.tie_word_embeddings());
}

void test_valid_document()
{
    auto result = Qwen3Config::from_json(make_valid_document());
    assert(result.has_value());
    assert_tiny_config(*result);
}

void test_optional_fields_and_legacy_rope_theta()
{
    auto document = make_valid_document();
    document.erase("num_key_value_heads");
    document.erase("head_dim");
    document.erase("layer_types");
    document.erase("dtype");
    document.erase("use_sliding_window");
    document.erase("sliding_window");
    document.erase("rope_parameters");
    document["rope_theta"] = 10'000.0;

    auto result = Qwen3Config::from_json(document);
    assert(result.has_value());
    assert(result->num_key_value_heads() == 4);
    assert(result->head_dim() == 32);
    assert(result->rope_theta() == 10'000.0F);
}

void test_invalid_documents()
{
    {
        auto document = make_valid_document();
        document["model_type"] = "qwen2";
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["hidden_act"] = "gelu";
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["num_key_value_heads"] = 3;
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["head_dim"] = 31;
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["dtype"] = "bfloat16";
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["layer_types"][1] = "sliding_attention";
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["use_sliding_window"] = true;
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["rms_norm_eps"] = 0.0;
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["rope_parameters"]["rope_type"] = "linear";
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document.erase("vocab_size");
        assert_invalid_config(document);
    }
    {
        auto document = make_valid_document();
        document["hidden_size"] = -1;
        assert_invalid_config(document);
    }
}

void test_load_from_file()
{
    TemporaryFixture fixture;
    const auto path = fixture.write(make_valid_document().dump());

    auto filesystem = make_filesystem();
    auto result = Qwen3Config::load(filesystem, path);
    assert(result.has_value());
    assert_tiny_config(*result);
}

void test_file_errors()
{
    TemporaryFixture fixture;
    auto filesystem = make_filesystem();

    const auto missing_path = fixture.directory() / "missing.json";
    auto missing_result = Qwen3Config::load(filesystem, missing_path);
    assert(!missing_result.has_value());
    assert(missing_result.error().category() == common::ErrorCategory::Filesystem);

    const auto invalid_path = fixture.write("{");
    auto invalid_result = Qwen3Config::load(filesystem, invalid_path);
    assert(!invalid_result.has_value());
    assert(invalid_result.error().category() == common::ErrorCategory::Model);
    assert(
        invalid_result.error().code() == std::to_underlying(ModelErrorCode::InvalidConfiguration)
    );
}

} // namespace

int main()
{
    test_valid_document();
    test_optional_fields_and_legacy_rope_theta();
    test_invalid_documents();
    test_load_from_file();
    test_file_errors();
}
