#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
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
#include "core/runtime/qwen3/qwen3_text_generator.hpp"

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

struct TensorSpec
{
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

std::size_t shape_numel(const std::vector<std::size_t> & shape)
{
    std::size_t result = 1;
    for (const auto dimension : shape) {
        result *= dimension;
    }
    return result;
}

std::vector<float> zeros(const std::vector<std::size_t> & shape)
{
    return std::vector<float>(shape_numel(shape), 0.0F);
}

void append_byte(std::vector<std::byte> & output, unsigned char value)
{
    output.push_back(static_cast<std::byte>(value));
}

void append_u16_le(std::vector<std::byte> & output, std::uint16_t value)
{
    for (std::size_t shift = 0; shift < 16; shift += 8) {
        append_byte(output, static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}

void append_u64_le(std::vector<std::byte> & output, std::uint64_t value)
{
    for (std::size_t shift = 0; shift < 64; shift += 8) {
        append_byte(output, static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}

void append_bfloat16_le(std::vector<std::byte> & output, float value)
{
    const auto float32_bits = std::bit_cast<std::uint32_t>(value);
    append_u16_le(output, static_cast<std::uint16_t>(float32_bits >> 16U));
}

json::Document make_model_config()
{
    return json::Document::parse(
        R"({
            "model_type": "qwen3",
            "hidden_act": "silu",
            "vocab_size": 4,
            "hidden_size": 4,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "rope_type": "default"
            },
            "attention_bias": false,
            "tie_word_embeddings": true,
            "dtype": "bfloat16",
            "layer_types": ["full_attention"],
            "use_sliding_window": false,
            "sliding_window": null
        })"
    );
}

std::vector<TensorSpec> make_model_tensors()
{
    std::vector<TensorSpec> tensors;
    tensors.push_back(
        TensorSpec {
            "model.embed_tokens.weight",
            {4, 4},
            {1.0F,
             0.0F,
             0.0F,
             0.0F,
             0.0F,
             1.0F,
             0.0F,
             0.0F,
             0.0F,
             0.0F,
             1.0F,
             0.0F,
             0.0F,
             0.0F,
             0.0F,
             1.0F}
        }
    );

    tensors.push_back(TensorSpec {"model.layers.0.self_attn.q_proj.weight", {4, 4}, zeros({4, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.self_attn.k_proj.weight", {2, 4}, zeros({2, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.self_attn.v_proj.weight", {2, 4}, zeros({2, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.self_attn.o_proj.weight", {4, 4}, zeros({4, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.self_attn.q_norm.weight", {2}, {1.0F, 1.0F}});
    tensors.push_back(TensorSpec {"model.layers.0.self_attn.k_norm.weight", {2}, {1.0F, 1.0F}});
    tensors.push_back(TensorSpec {"model.layers.0.mlp.gate_proj.weight", {4, 4}, zeros({4, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.mlp.up_proj.weight", {4, 4}, zeros({4, 4})});
    tensors.push_back(TensorSpec {"model.layers.0.mlp.down_proj.weight", {4, 4}, zeros({4, 4})});
    tensors.push_back(
        TensorSpec {"model.layers.0.input_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}}
    );
    tensors.push_back(
        TensorSpec {"model.layers.0.post_attention_layernorm.weight", {4}, {1.0F, 1.0F, 1.0F, 1.0F}}
    );
    tensors.push_back(TensorSpec {"model.norm.weight", {4}, {1.5F, 1.0F, 1.0F, 1.0F}});
    return tensors;
}

void write_safetensors(const std::filesystem::path & path, const std::vector<TensorSpec> & tensors)
{
    json::Document header = json::Document::object();
    std::vector<std::byte> data;
    std::uint64_t data_offset = 0;

    for (const auto & tensor : tensors) {
        constexpr std::size_t element_size = sizeof(std::uint16_t);
        const auto tensor_bytes = static_cast<std::uint64_t>(tensor.values.size() * element_size);

        json::Document descriptor;
        descriptor["dtype"] = "BF16";
        descriptor["shape"] = tensor.shape;
        descriptor["data_offsets"] = {data_offset, data_offset + tensor_bytes};
        header[tensor.name] = std::move(descriptor);

        for (const auto value : tensor.values) {
            append_bfloat16_le(data, value);
        }
        data_offset += tensor_bytes;
    }

    const auto header_text = header.dump();
    std::vector<std::byte> file_bytes;
    file_bytes.reserve(sizeof(std::uint64_t) + header_text.size() + data.size());
    append_u64_le(file_bytes, static_cast<std::uint64_t>(header_text.size()));
    for (const auto character : header_text) {
        append_byte(file_bytes, static_cast<unsigned char>(character));
    }
    file_bytes.insert(file_bytes.end(), data.begin(), data.end());

    std::ofstream stream(path, std::ios::binary);
    assert(stream.is_open());
    stream.write(
        reinterpret_cast<const char *>(file_bytes.data()),
        static_cast<std::streamsize>(file_bytes.size())
    );
    assert(stream.good());
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
    return document;
}

json::Document make_tokenizer_config()
{
    json::Document config = json::Document::object();
    config["bos_token"] = nullptr;
    config["eos_token"] = "a";
    config["pad_token"] = nullptr;
    return config;
}

class TemporaryModel
{
public:
    TemporaryModel()
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_qwen3_text_generator_test_" +
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

        std::ofstream config_stream(directory_ / "config.json");
        assert(config_stream.is_open());
        config_stream << make_model_config().dump();
        assert(config_stream.good());

        write_safetensors(directory_ / "model.safetensors", make_model_tensors());

        write("tokenizer.json", make_tokenizer_document());
        write("tokenizer_config.json", make_tokenizer_config());
    }

    TemporaryModel(const TemporaryModel &) = delete;
    TemporaryModel & operator=(const TemporaryModel &) = delete;

    ~TemporaryModel()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    [[nodiscard]]
    const std::filesystem::path & directory() const noexcept
    {
        return directory_;
    }

private:
    void write(std::string_view name, const json::Document & document) const
    {
        std::ofstream stream(directory_ / name);
        assert(stream.is_open());
        stream << document.dump();
        assert(stream.good());
    }

private:
    std::filesystem::path directory_;
};

filesystem::Filesystem make_filesystem()
{
    auto result = filesystem::Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    assert(result.has_value());
    return std::move(*result);
}

void test_single_turn_text_generation()
{
    TemporaryModel fixture;
    auto filesystem = make_filesystem();

    auto generator_result = Qwen3TextGenerator::load(filesystem, fixture.directory());
    assert(generator_result.has_value());
    auto generator = std::move(*generator_result);

    const auto generated = generator.generate("ab", 2);
    assert(generated.has_value());
    assert(*generated == "abab");

    const auto no_new_tokens = generator.generate("ab", 0);
    assert(no_new_tokens.has_value());
    assert(no_new_tokens->empty());

    const auto empty_prompt = generator.generate("", 1);
    assert(!empty_prompt.has_value());
    assert(empty_prompt.error().category() == common::ErrorCategory::Model);
}

} // namespace

int main()
{
    test_single_turn_text_generation();
    return 0;
}
