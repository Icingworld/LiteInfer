#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/error.hpp"
#include "core/filesystem/filesystem.hpp"
#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif
#include "core/json/json.hpp"
#include "core/model/model_error.hpp"
#include "core/model/qwen3/qwen3_generator.hpp"
#include "core/model/qwen3/qwen3_model.hpp"
#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core;
using model::ModelErrorCode;
using model::qwen3::Qwen3Generator;
using model::qwen3::Qwen3Model;

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

void append_u32_le(std::vector<std::byte> & output, std::uint32_t value)
{
    for (std::size_t shift = 0; shift < 32; shift += 8) {
        append_byte(output, static_cast<unsigned char>((value >> shift) & 0xffU));
    }
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

void append_float_le(std::vector<std::byte> & output, float value)
{
    append_u32_le(output, std::bit_cast<std::uint32_t>(value));
}

void append_bfloat16_le(std::vector<std::byte> & output, float value)
{
    const auto float32_bits = std::bit_cast<std::uint32_t>(value);
    append_u16_le(output, static_cast<std::uint16_t>(float32_bits >> 16U));
}

json::Document make_config()
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
            "dtype": "float32",
            "layer_types": ["full_attention"],
            "use_sliding_window": false,
            "sliding_window": null
        })"
    );
}

std::vector<TensorSpec> make_tensors(bool use_bfloat16 = false, bool use_nonzero_attention = false)
{
    auto q_proj_weights = zeros({4, 4});
    auto k_proj_weights = zeros({2, 4});
    auto v_proj_weights = zeros({2, 4});
    auto o_proj_weights = zeros({4, 4});
    if (use_nonzero_attention) {
        q_proj_weights = {
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
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        };
        k_proj_weights = {
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
        };
        v_proj_weights = {
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        };
        o_proj_weights = {
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
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
        };
    }

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

    tensors.push_back(
        TensorSpec {
            "model.layers.0.self_attn.q_proj.weight",
            {4, 4},
            std::move(q_proj_weights),
        }
    );
    tensors.push_back(
        TensorSpec {
            "model.layers.0.self_attn.k_proj.weight",
            {2, 4},
            std::move(k_proj_weights),
        }
    );
    tensors.push_back(
        TensorSpec {
            "model.layers.0.self_attn.v_proj.weight",
            {2, 4},
            std::move(v_proj_weights),
        }
    );
    tensors.push_back(
        TensorSpec {
            "model.layers.0.self_attn.o_proj.weight",
            {4, 4},
            std::move(o_proj_weights),
        }
    );
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
    tensors.push_back(
        TensorSpec {"model.norm.weight", {4}, {use_bfloat16 ? 1.5F : 1.0F, 1.0F, 1.0F, 1.0F}}
    );
    return tensors;
}

void write_safetensors(
    const std::filesystem::path & path,
    const std::vector<TensorSpec> & tensors,
    bool use_bfloat16 = false
)
{
    json::Document header = json::Document::object();
    std::vector<std::byte> data;
    std::uint64_t data_offset = 0;

    for (const auto & tensor : tensors) {
        const std::size_t element_size = use_bfloat16 ? sizeof(std::uint16_t) : sizeof(float);
        const std::uint64_t tensor_bytes =
            static_cast<std::uint64_t>(tensor.values.size() * element_size);

        json::Document descriptor;
        descriptor["dtype"] = use_bfloat16 ? "BF16" : "F32";
        descriptor["shape"] = tensor.shape;
        descriptor["data_offsets"] = {data_offset, data_offset + tensor_bytes};
        header[tensor.name] = std::move(descriptor);

        for (const auto value : tensor.values) {
            if (use_bfloat16) {
                append_bfloat16_le(data, value);
            } else {
                append_float_le(data, value);
            }
        }
        data_offset += tensor_bytes;
    }

    const auto header_text = header.dump();
    std::vector<std::byte> file_bytes;
    file_bytes.reserve(sizeof(std::uint64_t) + header_text.size() + data.size());
    append_u64_le(file_bytes, static_cast<std::uint64_t>(header_text.size()));
    for (const char character : header_text) {
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

class TemporaryModel
{
public:
    explicit TemporaryModel(
        bool use_bfloat16 = false,
        bool use_nonzero_attention = false
    )
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_qwen3_model_test_" +
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
        auto config = make_config();
        if (use_bfloat16) {
            config["dtype"] = "bfloat16";
        }
        config_stream << config.dump();
        assert(config_stream.good());

        write_safetensors(
            directory_ / "model.safetensors",
            make_tensors(use_bfloat16, use_nonzero_attention),
            use_bfloat16
        );
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
    std::filesystem::path directory_;
};

filesystem::Filesystem make_filesystem()
{
    auto result = filesystem::Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    assert(result.has_value());
    return std::move(*result);
}

tensor::Tensor make_token_ids(std::initializer_list<std::int64_t> values)
{
    auto result = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Int64,
        tensor::Shape {values.size()}
    );
    assert(result.has_value());

    auto data = result->data_as<std::int64_t>();
    assert(data.has_value());
    std::size_t index = 0;
    for (const auto value : values) {
        (*data)[index++] = value;
    }
    return std::move(*result);
}

void assert_shape(const tensor::Tensor & value, std::initializer_list<std::size_t> expected)
{
    const auto actual = value.shape().values();
    assert(actual.size() == expected.size());
    std::size_t index = 0;
    for (const auto dimension : expected) {
        assert(actual[index++] == dimension);
    }
}

void assert_near(float actual, float expected, float tolerance = 1.0e-5F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_prefill_decode_matches_full_forward()
{
    TemporaryModel fixture(false, true);
    auto filesystem = make_filesystem();
    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());
    auto & model = *model_result;

    auto prompt = make_token_ids({0, 1});
    auto full_prompt = model.forward(prompt);
    assert(full_prompt.has_value());
    assert_shape(*full_prompt, {2, 4});
    auto full_prompt_values = full_prompt->data_as<float>();
    assert(full_prompt_values.has_value());

    auto cache_result = model.create_kv_cache();
    assert(cache_result.has_value());
    auto & cache = *cache_result;

    auto prefill = model.prefill(prompt, cache);
    assert(prefill.has_value());
    assert_shape(*prefill, {1, 4});
    assert(cache.length() == 2);
    auto prefill_values = prefill->data_as<float>();
    assert(prefill_values.has_value());
    for (std::size_t index = 0; index < 4; ++index) {
        assert_near((*prefill_values)[index], (*full_prompt_values)[4 + index]);
    }

    auto next_token = make_token_ids({2});
    auto full_sequence = make_token_ids({0, 1, 2});
    auto full_output = model.forward(full_sequence);
    assert(full_output.has_value());
    assert_shape(*full_output, {3, 4});
    auto full_output_values = full_output->data_as<float>();
    assert(full_output_values.has_value());

    auto decoded = model.decode(next_token, cache);
    assert(decoded.has_value());
    assert_shape(*decoded, {1, 4});
    assert(cache.length() == 3);
    auto decoded_values = decoded->data_as<float>();
    assert(decoded_values.has_value());
    for (std::size_t index = 0; index < 4; ++index) {
        assert_near((*decoded_values)[index], (*full_output_values)[8 + index]);
    }

    auto continuation = make_token_ids({3, 0});
    auto extended_sequence = make_token_ids({0, 1, 2, 3, 0});
    auto extended_output = model.forward(extended_sequence);
    assert(extended_output.has_value());
    assert_shape(*extended_output, {5, 4});
    auto extended_output_values = extended_output->data_as<float>();
    assert(extended_output_values.has_value());

    auto continued_prefill = model.prefill(continuation, cache);
    assert(continued_prefill.has_value());
    assert_shape(*continued_prefill, {1, 4});
    assert(cache.length() == 5);
    auto continued_prefill_values = continued_prefill->data_as<float>();
    assert(continued_prefill_values.has_value());
    for (std::size_t index = 0; index < 4; ++index) {
        assert_near((*continued_prefill_values)[index], (*extended_output_values)[16 + index]);
    }

    auto final_token = make_token_ids({1});
    auto final_sequence = make_token_ids({0, 1, 2, 3, 0, 1});
    auto final_output = model.forward(final_sequence);
    assert(final_output.has_value());
    assert_shape(*final_output, {6, 4});
    auto final_output_values = final_output->data_as<float>();
    assert(final_output_values.has_value());

    auto final_decoded = model.decode(final_token, cache);
    assert(final_decoded.has_value());
    assert_shape(*final_decoded, {1, 4});
    assert(cache.length() == 6);
    auto final_decoded_values = final_decoded->data_as<float>();
    assert(final_decoded_values.has_value());
    for (std::size_t index = 0; index < 4; ++index) {
        assert_near((*final_decoded_values)[index], (*final_output_values)[20 + index]);
    }
}

void test_cached_validation_and_exact_capacity()
{
    TemporaryModel fixture;
    auto filesystem = make_filesystem();
    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());
    auto & model = *model_result;

    auto cache_result = model.create_kv_cache();
    assert(cache_result.has_value());
    auto & cache = *cache_result;

    auto invalid_decode = model.decode(make_token_ids({0, 1}), cache);
    assert(!invalid_decode.has_value());
    assert(invalid_decode.error().category() == common::ErrorCategory::Model);
    assert(cache.length() == 0);

    auto invalid_token = model.prefill(make_token_ids({4}), cache);
    assert(!invalid_token.has_value());
    assert(invalid_token.error().category() == common::ErrorCategory::Embedding);
    assert(cache.length() == 0);

    auto seven_tokens = make_token_ids({0, 1, 2, 3, 0, 1, 2});
    auto seven_result = model.prefill(seven_tokens, cache);
    assert(seven_result.has_value());
    assert(cache.length() == 7);

    auto eighth_result = model.decode(make_token_ids({3}), cache);
    assert(eighth_result.has_value());
    assert(cache.length() == 8);

    auto beyond_capacity = model.decode(make_token_ids({0}), cache);
    assert(!beyond_capacity.has_value());
    assert(beyond_capacity.error().category() == common::ErrorCategory::Model);
    assert(cache.length() == 8);
}

void test_load_and_forward()
{
    TemporaryModel fixture;
    auto filesystem = make_filesystem();

    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());
    auto & model = *model_result;
    assert(model.vocab_size() == 4);
    assert(model.hidden_size() == 4);
    assert(model.num_hidden_layers() == 1);
    assert(model.config().num_key_value_heads() == 1);

    auto input = make_token_ids({0, 1});
    auto output = model.forward(input);
    assert(output.has_value());
    assert_shape(*output, {2, 4});

    auto values = output->data_as<float>();
    assert(values.has_value());
    const float expected = 1.0F / std::sqrt(0.25F + 1.0e-6F);
    assert_near((*values)[0], expected);
    assert_near((*values)[1], 0.0F);
    assert_near((*values)[2], 0.0F);
    assert_near((*values)[3], 0.0F);
    assert_near((*values)[4], 0.0F);
    assert_near((*values)[5], expected);
    assert_near((*values)[6], 0.0F);
    assert_near((*values)[7], 0.0F);

    auto batch_input = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Int32,
        tensor::Shape {1, 2}
    );
    assert(batch_input.has_value());
    auto batch_values = batch_input->data_as<std::int32_t>();
    assert(batch_values.has_value());
    (*batch_values)[0] = 2;
    (*batch_values)[1] = 3;

    auto batch_output = model.forward(*batch_input);
    assert(batch_output.has_value());
    assert_shape(*batch_output, {1, 2, 4});
}

void test_bfloat16_weights_are_converted_to_float32()
{
    TemporaryModel fixture(true);
    auto filesystem = make_filesystem();

    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());

    auto output = model_result->forward(make_token_ids({0, 1}));
    assert(output.has_value());
    assert(output->data_type() == liteinfer::core::common::data_type::DataType::Float32);
    assert_shape(*output, {2, 4});

    auto values = output->data_as<float>();
    assert(values.has_value());
    const float expected = 1.5F / std::sqrt(0.25F + 1.0e-6F);
    assert_near((*values)[0], expected);
    assert_near((*values)[1], 0.0F);
    assert_near((*values)[2], 0.0F);
    assert_near((*values)[3], 0.0F);
}

void test_forward_validation()
{
    TemporaryModel fixture;
    auto filesystem = make_filesystem();
    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());

    auto too_long = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Int64,
        tensor::Shape {9}
    );
    assert(too_long.has_value());
    auto too_long_result = model_result->forward(*too_long);
    assert(!too_long_result.has_value());
    assert(too_long_result.error().category() == common::ErrorCategory::Model);
    assert(too_long_result.error().code() == std::to_underlying(ModelErrorCode::InvalidInput));

    auto invalid_token = make_token_ids({4});
    auto invalid_token_result = model_result->forward(invalid_token);
    assert(!invalid_token_result.has_value());
    assert(invalid_token_result.error().category() == common::ErrorCategory::Embedding);
}

void test_greedy_generation()
{
    TemporaryModel fixture;
    auto filesystem = make_filesystem();
    auto model_result = Qwen3Model::load(filesystem, fixture.directory());
    assert(model_result.has_value());

    Qwen3Generator generator(*model_result);
    const std::vector<common::TokenId> prompt {0};

    auto generated = generator.generate(prompt, 3);
    assert(generated.has_value());
    assert(*generated == std::vector<common::TokenId>({0, 0, 0, 0}));

    auto eos_generated = generator.generate(prompt, 5, 0);
    assert(eos_generated.has_value());
    assert(*eos_generated == std::vector<common::TokenId>({0, 0}));

    auto no_new_tokens = generator.generate(prompt, 0);
    assert(no_new_tokens.has_value());
    assert(*no_new_tokens == prompt);

    const std::vector<common::TokenId> empty_prompt;
    auto empty_result = generator.generate(empty_prompt, 1);
    assert(!empty_result.has_value());
    assert(empty_result.error().category() == common::ErrorCategory::Model);
    assert(empty_result.error().code() == std::to_underlying(ModelErrorCode::InvalidInput));

    auto too_many = generator.generate(prompt, 8);
    assert(!too_many.has_value());
    assert(too_many.error().code() == std::to_underlying(ModelErrorCode::InvalidInput));
}

void test_missing_weights()
{
    TemporaryModel fixture;
    std::filesystem::remove(fixture.directory() / "model.safetensors");

    auto filesystem = make_filesystem();
    auto result = Qwen3Model::load(filesystem, fixture.directory());
    assert(!result.has_value());
    assert(result.error().category() == common::ErrorCategory::Filesystem);
}

} // namespace

int main()
{
    test_prefill_decode_matches_full_forward();
    test_cached_validation_and_exact_capacity();
    test_load_and_forward();
    test_bfloat16_weights_are_converted_to_float32();
    test_forward_validation();
    test_greedy_generation();
    test_missing_weights();
}
