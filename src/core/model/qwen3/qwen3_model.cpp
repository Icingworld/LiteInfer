#include "core/model/qwen3/qwen3_model.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/safetensors/safetensors_file.hpp"

namespace liteinfer::core::model::qwen3
{

namespace
{

constexpr std::string_view CONFIG_FILE_NAME = "config.json";
constexpr std::string_view WEIGHTS_FILE_NAME = "model.safetensors";

ModelError invalid_configuration(std::string_view message)
{
    return ModelError(ModelErrorCode::InvalidConfiguration, message);
}

ModelError invalid_tensor(std::string_view name, std::string_view reason)
{
    std::string message;
    message.reserve(name.size() + reason.size() + 32);
    message += "Qwen3 model tensor '";
    message += name;
    message += "' ";
    message += reason;
    return invalid_configuration(message);
}

std::expected<std::vector<std::byte>, ModelError> convert_bfloat16_to_float32(
    std::span<const std::byte> bytes,
    std::size_t numel,
    std::string_view name
)
{
    if (numel > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) ||
        bytes.size() != numel * sizeof(std::uint16_t)) [[unlikely]] {
        return std::unexpected(invalid_tensor(name, "has an invalid BF16 byte size"));
    }
    if (numel > std::numeric_limits<std::size_t>::max() / sizeof(float)) [[unlikely]] {
        return std::unexpected(invalid_tensor(name, "is too large to convert to Float32"));
    }

    std::vector<std::byte> converted(numel * sizeof(float));
    for (std::size_t index = 0; index < numel; ++index) {
        const std::size_t byte_offset = index * sizeof(std::uint16_t);
        const auto low = std::to_integer<std::uint8_t>(bytes[byte_offset]);
        const auto high = std::to_integer<std::uint8_t>(bytes[byte_offset + 1]);
        const auto bfloat16_bits =
            static_cast<std::uint16_t>(low) | (static_cast<std::uint16_t>(high) << 8U);
        const auto float32_bits = static_cast<std::uint32_t>(bfloat16_bits) << 16U;
        const float value = std::bit_cast<float>(float32_bits);
        std::memcpy(converted.data() + index * sizeof(float), &value, sizeof(value));
    }
    return converted;
}

std::expected<tensor::Tensor, ModelError> load_float32_tensor(
    safetensors::SafetensorsFile & weights,
    std::string_view name,
    const std::vector<std::size_t> & expected_shape
)
{
    auto descriptor = weights.tensor_info(name);
    if (!descriptor) [[unlikely]] {
        return std::unexpected(std::move(descriptor).error());
    }

    if (descriptor->dtype != "F32" && descriptor->dtype != "BF16") [[unlikely]] {
        return std::unexpected(invalid_tensor(name, "must use F32 or BF16 dtype"));
    }

    if (descriptor->shape.size() != expected_shape.size()) [[unlikely]] {
        return std::unexpected(invalid_tensor(name, "has an unexpected rank"));
    }

    for (std::size_t dimension = 0; dimension < expected_shape.size(); ++dimension) {
        if (descriptor->shape[dimension] > std::numeric_limits<std::size_t>::max() ||
            static_cast<std::size_t>(descriptor->shape[dimension]) != expected_shape[dimension])
            [[unlikely]] {
            return std::unexpected(invalid_tensor(name, "has an unexpected shape"));
        }
    }

    auto shape = tensor::Shape(expected_shape);
    auto numel = shape.numel();
    if (!numel) [[unlikely]] {
        return std::unexpected(std::move(numel).error());
    }

    auto bytes = weights.read_tensor(name);
    if (!bytes) [[unlikely]] {
        return std::unexpected(std::move(bytes).error());
    }

    std::vector<std::byte> converted;
    std::span<const std::byte> float32_bytes = *bytes;
    if (descriptor->dtype == "BF16") {
        auto conversion = convert_bfloat16_to_float32(*bytes, *numel, name);
        if (!conversion) [[unlikely]] {
            return std::unexpected(std::move(conversion).error());
        }
        converted = std::move(*conversion);
        float32_bytes = converted;
    }

    auto result =
        tensor::Tensor::from_bytes(tensor::DataType::Float32, std::move(shape), float32_bytes);
    if (!result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    return std::move(*result);
}

std::expected<layer::Linear, ModelError> load_linear(
    safetensors::SafetensorsFile & weights,
    std::string_view weight_name,
    const std::vector<std::size_t> & weight_shape,
    const std::string * bias_name,
    const std::vector<std::size_t> & bias_shape
)
{
    auto weight = load_float32_tensor(weights, weight_name, weight_shape);
    if (!weight) [[unlikely]] {
        return std::unexpected(std::move(weight).error());
    }

    std::optional<tensor::Tensor> bias;
    if (bias_name != nullptr) {
        auto bias_tensor = load_float32_tensor(weights, *bias_name, bias_shape);
        if (!bias_tensor) [[unlikely]] {
            return std::unexpected(std::move(bias_tensor).error());
        }
        bias.emplace(std::move(*bias_tensor));
    }

    auto result = layer::Linear::create(std::move(*weight), std::move(bias));
    if (!result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    return std::move(*result);
}

std::expected<layer::RMSNorm, ModelError> load_rms_norm(
    safetensors::SafetensorsFile & weights,
    std::string_view weight_name,
    std::size_t normalized_size,
    float eps
)
{
    auto weight = load_float32_tensor(weights, weight_name, {normalized_size});
    if (!weight) [[unlikely]] {
        return std::unexpected(std::move(weight).error());
    }

    auto result = layer::RMSNorm::create(std::move(*weight), eps);
    if (!result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    return std::move(*result);
}

} // namespace

Qwen3Model::Qwen3Model(
    Qwen3Config config,
    embedding::Embedding embed_tokens,
    std::vector<Qwen3DecoderLayer> layers,
    layer::RMSNorm norm,
    layer::Linear lm_head
)
    : config_(std::move(config))
    , embed_tokens_(std::move(embed_tokens))
    , layers_(std::move(layers))
    , norm_(std::move(norm))
    , lm_head_(std::move(lm_head))
{}

std::expected<Qwen3Model, ModelError>
Qwen3Model::load(filesystem::Filesystem & filesystem, const std::filesystem::path & model_directory)
{
    auto config_result = Qwen3Config::load(filesystem, model_directory / CONFIG_FILE_NAME);
    if (!config_result) [[unlikely]] {
        return std::unexpected(std::move(config_result).error());
    }
    Qwen3Config config = std::move(*config_result);

    auto weights_result =
        safetensors::SafetensorsFile::open(filesystem, model_directory / WEIGHTS_FILE_NAME);
    if (!weights_result) [[unlikely]] {
        return std::unexpected(std::move(weights_result).error());
    }
    auto weights = std::move(*weights_result);

    const std::size_t query_width = config.num_attention_heads() * config.head_dim();
    const std::size_t key_value_width = config.num_key_value_heads() * config.head_dim();

    auto embedding_weight = load_float32_tensor(
        weights,
        "model.embed_tokens.weight",
        {config.vocab_size(), config.hidden_size()}
    );
    if (!embedding_weight) [[unlikely]] {
        return std::unexpected(std::move(embedding_weight).error());
    }

    // Embedding 和 tied lm_head 需要各自持有一份 Tensor。Tensor 的默认拷贝是深拷贝，
    // 因此先用拷贝创建 embedding，再移动原始权重给 lm_head。
    auto embedding_result = embedding::Embedding::create(*embedding_weight);
    if (!embedding_result) [[unlikely]] {
        return std::unexpected(std::move(embedding_result).error());
    }
    auto embed_tokens = std::move(*embedding_result);

    auto lm_head_result = [&]() -> std::expected<layer::Linear, ModelError> {
        if (config.tie_word_embeddings()) {
            return layer::Linear::create(std::move(*embedding_weight));
        }
        return load_linear(
            weights,
            "lm_head.weight",
            {config.vocab_size(), config.hidden_size()},
            nullptr,
            {}
        );
    }();
    if (!lm_head_result) [[unlikely]] {
        return std::unexpected(std::move(lm_head_result).error());
    }
    auto lm_head = std::move(*lm_head_result);

    std::vector<Qwen3DecoderLayer> layers;
    layers.reserve(config.num_hidden_layers());
    for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers(); ++layer_index) {
        const std::string prefix = "model.layers." + std::to_string(layer_index);
        const std::string q_proj_weight_name = prefix + ".self_attn.q_proj.weight";
        const std::string k_proj_weight_name = prefix + ".self_attn.k_proj.weight";
        const std::string v_proj_weight_name = prefix + ".self_attn.v_proj.weight";
        const std::string o_proj_weight_name = prefix + ".self_attn.o_proj.weight";
        const std::string q_proj_bias_name = prefix + ".self_attn.q_proj.bias";
        const std::string k_proj_bias_name = prefix + ".self_attn.k_proj.bias";
        const std::string v_proj_bias_name = prefix + ".self_attn.v_proj.bias";
        const std::string o_proj_bias_name = prefix + ".self_attn.o_proj.bias";

        const std::string q_norm_name = prefix + ".self_attn.q_norm.weight";
        const std::string k_norm_name = prefix + ".self_attn.k_norm.weight";
        const std::string gate_proj_name = prefix + ".mlp.gate_proj.weight";
        const std::string up_proj_name = prefix + ".mlp.up_proj.weight";
        const std::string down_proj_name = prefix + ".mlp.down_proj.weight";
        const std::string input_norm_name = prefix + ".input_layernorm.weight";
        const std::string post_attention_norm_name = prefix + ".post_attention_layernorm.weight";

        const std::string * q_bias = config.attention_bias() ? &q_proj_bias_name : nullptr;
        const std::string * k_bias = config.attention_bias() ? &k_proj_bias_name : nullptr;
        const std::string * v_bias = config.attention_bias() ? &v_proj_bias_name : nullptr;
        const std::string * o_bias = config.attention_bias() ? &o_proj_bias_name : nullptr;

        auto q_proj = load_linear(
            weights,
            q_proj_weight_name,
            {query_width, config.hidden_size()},
            q_bias,
            {query_width}
        );
        if (!q_proj) [[unlikely]] {
            return std::unexpected(std::move(q_proj).error());
        }
        auto k_proj = load_linear(
            weights,
            k_proj_weight_name,
            {key_value_width, config.hidden_size()},
            k_bias,
            {key_value_width}
        );
        if (!k_proj) [[unlikely]] {
            return std::unexpected(std::move(k_proj).error());
        }
        auto v_proj = load_linear(
            weights,
            v_proj_weight_name,
            {key_value_width, config.hidden_size()},
            v_bias,
            {key_value_width}
        );
        if (!v_proj) [[unlikely]] {
            return std::unexpected(std::move(v_proj).error());
        }
        auto o_proj = load_linear(
            weights,
            o_proj_weight_name,
            {config.hidden_size(), query_width},
            o_bias,
            {config.hidden_size()}
        );
        if (!o_proj) [[unlikely]] {
            return std::unexpected(std::move(o_proj).error());
        }

        auto q_norm = load_rms_norm(weights, q_norm_name, config.head_dim(), config.rms_norm_eps());
        if (!q_norm) [[unlikely]] {
            return std::unexpected(std::move(q_norm).error());
        }
        auto k_norm = load_rms_norm(weights, k_norm_name, config.head_dim(), config.rms_norm_eps());
        if (!k_norm) [[unlikely]] {
            return std::unexpected(std::move(k_norm).error());
        }

        auto attention = Qwen3Attention::create(
            std::move(*q_proj),
            std::move(*k_proj),
            std::move(*v_proj),
            std::move(*o_proj),
            std::move(*q_norm),
            std::move(*k_norm),
            config.num_attention_heads(),
            config.num_key_value_heads(),
            config.rope_theta()
        );
        if (!attention) [[unlikely]] {
            return std::unexpected(std::move(attention).error());
        }

        auto gate_proj = load_linear(
            weights,
            gate_proj_name,
            {config.intermediate_size(), config.hidden_size()},
            nullptr,
            {}
        );
        if (!gate_proj) [[unlikely]] {
            return std::unexpected(std::move(gate_proj).error());
        }
        auto up_proj = load_linear(
            weights,
            up_proj_name,
            {config.intermediate_size(), config.hidden_size()},
            nullptr,
            {}
        );
        if (!up_proj) [[unlikely]] {
            return std::unexpected(std::move(up_proj).error());
        }
        auto down_proj = load_linear(
            weights,
            down_proj_name,
            {config.hidden_size(), config.intermediate_size()},
            nullptr,
            {}
        );
        if (!down_proj) [[unlikely]] {
            return std::unexpected(std::move(down_proj).error());
        }

        auto mlp =
            Qwen3MLP::create(std::move(*gate_proj), std::move(*up_proj), std::move(*down_proj));
        if (!mlp) [[unlikely]] {
            return std::unexpected(std::move(mlp).error());
        }

        auto input_norm =
            load_rms_norm(weights, input_norm_name, config.hidden_size(), config.rms_norm_eps());
        if (!input_norm) [[unlikely]] {
            return std::unexpected(std::move(input_norm).error());
        }
        auto post_attention_norm = load_rms_norm(
            weights,
            post_attention_norm_name,
            config.hidden_size(),
            config.rms_norm_eps()
        );
        if (!post_attention_norm) [[unlikely]] {
            return std::unexpected(std::move(post_attention_norm).error());
        }

        auto decoder_layer = Qwen3DecoderLayer::create(
            std::move(*attention),
            std::move(*mlp),
            std::move(*input_norm),
            std::move(*post_attention_norm)
        );
        if (!decoder_layer) [[unlikely]] {
            return std::unexpected(std::move(decoder_layer).error());
        }
        layers.push_back(std::move(*decoder_layer));
    }

    auto norm =
        load_rms_norm(weights, "model.norm.weight", config.hidden_size(), config.rms_norm_eps());
    if (!norm) [[unlikely]] {
        return std::unexpected(std::move(norm).error());
    }

    return Qwen3Model(
        std::move(config),
        std::move(embed_tokens),
        std::move(layers),
        std::move(*norm),
        std::move(lm_head)
    );
}

std::expected<tensor::Tensor, ModelError> Qwen3Model::forward(
    const tensor::Tensor & token_ids
) const
{
    if (token_ids.rank() != 1 && token_ids.rank() != 2) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3Model token IDs must have shape [sequence_length] or "
            "[batch_size, sequence_length]"
        ));
    }

    const auto input_shape = token_ids.shape().values();
    const std::size_t sequence_length = input_shape.back();
    if (sequence_length > config_.max_position_embeddings()) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3Model input sequence exceeds max_position_embeddings"
        ));
    }

    auto hidden_states = embed_tokens_.forward(token_ids);
    if (!hidden_states) [[unlikely]] {
        return std::unexpected(std::move(hidden_states).error());
    }
    tensor::Tensor hidden = std::move(*hidden_states);

    for (const auto & layer : layers_) {
        auto layer_output = layer.forward(hidden);
        if (!layer_output) [[unlikely]] {
            return std::unexpected(std::move(layer_output).error());
        }
        hidden = std::move(*layer_output);
    }

    auto normalized = norm_.forward(hidden);
    if (!normalized) [[unlikely]] {
        return std::unexpected(std::move(normalized).error());
    }

    auto logits = lm_head_.forward(*normalized);
    if (!logits) [[unlikely]] {
        return std::unexpected(std::move(logits).error());
    }
    return std::move(*logits);
}

const Qwen3Config & Qwen3Model::config() const noexcept
{
    return config_;
}

std::size_t Qwen3Model::vocab_size() const noexcept
{
    return config_.vocab_size();
}

std::size_t Qwen3Model::hidden_size() const noexcept
{
    return config_.hidden_size();
}

std::size_t Qwen3Model::num_hidden_layers() const noexcept
{
    return config_.num_hidden_layers();
}

} // namespace liteinfer::core::model::qwen3
