#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/error.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/json/json.hpp"
#include "core/tokenizer/qwen3/qwen3_tokenizer.hpp"
#include "core/tokenizer/tokenizer_error.hpp"

#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif

namespace
{

using namespace liteinfer::core;
using filesystem::Filesystem;
using tokenizer::TokenizerErrorCode;
using tokenizer::qwen3::Qwen3Tokenizer;

#if defined(_WIN32)
using NativeFilesystemBackend = filesystem::backend::windows::WindowsFilesystemBackend;
#else
using NativeFilesystemBackend = filesystem::backend::posix::PosixFilesystemBackend;
#endif

json::Document make_added_token(common::TokenId id, std::string_view content, bool special)
{
    json::Document token = json::Document::object();
    token["id"] = id;
    token["content"] = content;
    token["single_word"] = false;
    token["lstrip"] = false;
    token["rstrip"] = false;
    token["normalized"] = false;
    token["special"] = special;
    return token;
}

json::Document make_tokenizer_document()
{
    json::Document document = json::Document::object();

    document["normalizer"] = json::Document::object();
    document["normalizer"]["type"] = "NFC";

    document["pre_tokenizer"] = json::Document::object();
    document["pre_tokenizer"]["type"] = "Sequence";
    document["pre_tokenizer"]["pretokenizers"] = json::Document::array();

    json::Document split = json::Document::object();
    split["type"] = "Split";
    split["pattern"] = json::Document::object();
    split["pattern"]["Regex"] =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    split["behavior"] = "Isolated";
    document["pre_tokenizer"]["pretokenizers"].push_back(std::move(split));

    json::Document byte_level = json::Document::object();
    byte_level["type"] = "ByteLevel";
    byte_level["add_prefix_space"] = false;
    byte_level["use_regex"] = false;
    document["pre_tokenizer"]["pretokenizers"].push_back(std::move(byte_level));

    document["post_processor"] = json::Document::object();
    document["post_processor"]["type"] = "ByteLevel";
    document["decoder"] = json::Document::object();
    document["decoder"]["type"] = "ByteLevel";

    document["model"] = json::Document::object();
    document["model"]["type"] = "BPE";
    document["model"]["byte_fallback"] = false;
    document["model"]["vocab"] = json::Document::object();
    document["model"]["vocab"]["a"] = 0;
    document["model"]["vocab"]["b"] = 1;
    document["model"]["vocab"]["ab"] = 2;
    document["model"]["vocab"]["Ġ"] = 3;
    document["model"]["merges"] = json::Document::array();
    document["model"]["merges"].push_back("a b");

    document["added_tokens"] = json::Document::array();
    document["added_tokens"].push_back(make_added_token(4, "<special>", true));

    return document;
}

json::Document make_tokenizer_config()
{
    json::Document config = json::Document::object();
    config["added_tokens_decoder"] = json::Document::object();
    config["added_tokens_decoder"]["4"] = make_added_token(4, "<special>", true);
    config["added_tokens_decoder"]["5"] = make_added_token(5, "<plain>", false);
    config["bos_token"] = nullptr;
    config["eos_token"] = "<special>";
    config["pad_token"] = nullptr;
    return config;
}

class TemporaryFixture
{
public:
    TemporaryFixture()
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_qwen3_tokenizer_test_" +
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

    ~TemporaryFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        assert(!error);
    }

    void write(std::string_view name, const json::Document & document) const
    {
        std::ofstream stream(directory_ / name);
        assert(stream.is_open());
        stream << document.dump();
        assert(stream.good());
    }

    [[nodiscard]]
    const std::filesystem::path & directory() const noexcept
    {
        return directory_;
    }

private:
    std::filesystem::path directory_;
};

Filesystem make_filesystem()
{
    auto result = Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    assert(result.has_value());
    return std::move(*result);
}

void test_load_encode_decode()
{
    TemporaryFixture fixture;
    fixture.write("tokenizer.json", make_tokenizer_document());
    fixture.write("tokenizer_config.json", make_tokenizer_config());

    auto filesystem = make_filesystem();
    auto result = Qwen3Tokenizer::load(filesystem, fixture.directory());
    assert(result.has_value());
    auto tokenizer = std::move(*result);

    assert(tokenizer.vocab_size() == 4);
    assert(tokenizer.total_vocab_size() == 6);
    assert(!tokenizer.bos_token_id().has_value());
    assert(tokenizer.eos_token_id().has_value());
    assert(*tokenizer.eos_token_id() == 4);
    assert(!tokenizer.pad_token_id().has_value());
    assert(tokenizer.token_id("ab").has_value());
    assert(*tokenizer.token_id("ab") == 2);

    const auto encoded = tokenizer.encode("ab<special> a", false);
    assert(encoded.has_value());
    const std::vector<common::TokenId> expected {2, 4, 3, 0};
    assert(*encoded == expected);

    const auto decoded = tokenizer.decode(*encoded, true);
    assert(decoded.has_value());
    assert(*decoded == "ab a");

    const auto decoded_with_special = tokenizer.decode(*encoded, false);
    assert(decoded_with_special.has_value());
    assert(*decoded_with_special == "ab<special> a");

    const std::vector<common::TokenId> plain_added {5};
    const auto plain_decoded = tokenizer.decode(plain_added, true);
    assert(plain_decoded.has_value());
    assert(*plain_decoded == "<plain>");

    const std::vector<common::TokenId> unknown {6};
    const auto unknown_decoded = tokenizer.decode(unknown, true);
    assert(unknown_decoded.has_value());
    assert(unknown_decoded->empty());
}

void test_invalid_utf8()
{
    TemporaryFixture fixture;
    fixture.write("tokenizer.json", make_tokenizer_document());
    fixture.write("tokenizer_config.json", make_tokenizer_config());

    auto filesystem = make_filesystem();
    auto result = Qwen3Tokenizer::load(filesystem, fixture.directory());
    assert(result.has_value());

    const std::string invalid_utf8(1, static_cast<char>(0xc3));
    const auto encoded = result->encode(invalid_utf8);
    assert(!encoded.has_value());
    assert(encoded.error().category() == common::ErrorCategory::Tokenizer);
    assert(encoded.error().code() == std::to_underlying(TokenizerErrorCode::InvalidInput));
}

} // namespace

int main()
{
    test_load_encode_decode();
    test_invalid_utf8();
    return 0;
}
