#include "core/model/qwen3/qwen3_config.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/io/file_byte_reader.hpp"

namespace liteinfer::core::model::qwen3
{

namespace
{

constexpr std::uint64_t CONFIG_FILE_SIZE_LIMIT = 16ULL * 1024ULL * 1024ULL;

ModelError invalid_configuration(std::string_view message)
{
    return ModelError(ModelErrorCode::InvalidConfiguration, message);
}

ModelError invalid_field(std::string_view field, std::string_view reason)
{
    std::string message;
    message.reserve(field.size() + reason.size() + 32);
    message += "Qwen3 config field '";
    message += field;
    message += "' ";
    message += reason;
    return invalid_configuration(message);
}

std::expected<std::size_t, ModelError>
parse_positive_size(const json::Document & value, std::string_view field)
{
    std::uint64_t parsed_value = 0;
    if (value.is_number_unsigned()) {
        parsed_value = value.get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0) [[unlikely]] {
            return std::unexpected(invalid_field(field, "must be a positive integer"));
        }
        parsed_value = static_cast<std::uint64_t>(signed_value);
    } else [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a positive integer"));
    }

    if (parsed_value == 0) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a positive integer"));
    }
    if (parsed_value > std::numeric_limits<std::size_t>::max()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "does not fit in size_t"));
    }

    return static_cast<std::size_t>(parsed_value);
}

std::expected<std::size_t, ModelError>
read_positive_size(const json::Document & object, std::string_view field)
{
    const std::string key(field);
    if (!object.contains(key)) [[unlikely]] {
        return std::unexpected(invalid_field(field, "is missing"));
    }
    return parse_positive_size(object.at(key), field);
}

std::expected<float, ModelError>
parse_positive_float(const json::Document & value, std::string_view field)
{
    if (!value.is_number()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a positive number"));
    }

    const double parsed_value = value.get<double>();
    const float converted_value = static_cast<float>(parsed_value);
    if (!std::isfinite(parsed_value) || !std::isfinite(converted_value) || converted_value <= 0.0F)
        [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a finite positive number"));
    }

    return converted_value;
}

std::expected<float, ModelError>
read_positive_float(const json::Document & object, std::string_view field)
{
    const std::string key(field);
    if (!object.contains(key)) [[unlikely]] {
        return std::unexpected(invalid_field(field, "is missing"));
    }
    return parse_positive_float(object.at(key), field);
}

std::expected<std::string, ModelError>
read_required_string(const json::Document & object, std::string_view field)
{
    const std::string key(field);
    if (!object.contains(key)) [[unlikely]] {
        return std::unexpected(invalid_field(field, "is missing"));
    }

    const auto & value = object.at(key);
    if (!value.is_string()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a string"));
    }

    auto result = value.get<std::string>();
    if (result.empty()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must not be empty"));
    }
    return result;
}

std::expected<bool, ModelError>
read_required_boolean(const json::Document & object, std::string_view field)
{
    const std::string key(field);
    if (!object.contains(key)) [[unlikely]] {
        return std::unexpected(invalid_field(field, "is missing"));
    }

    const auto & value = object.at(key);
    if (!value.is_boolean()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a boolean"));
    }
    return value.get<bool>();
}

std::expected<float, ModelError> read_rope_theta(const json::Document & document)
{
    if (document.contains("rope_parameters")) {
        const auto & rope_parameters = document.at("rope_parameters");
        if (!rope_parameters.is_object()) [[unlikely]] {
            return std::unexpected(invalid_field("rope_parameters", "must be an object"));
        }

        if (rope_parameters.contains("rope_type")) {
            auto rope_type = read_required_string(rope_parameters, "rope_type");
            if (!rope_type) [[unlikely]] {
                return std::unexpected(std::move(rope_type).error());
            }
            if (*rope_type != "default") [[unlikely]] {
                return std::unexpected(
                    invalid_field("rope_parameters.rope_type", "must be 'default'")
                );
            }
        }

        if (!rope_parameters.contains("rope_theta")) [[unlikely]] {
            return std::unexpected(invalid_field("rope_parameters.rope_theta", "is missing"));
        }
        return parse_positive_float(rope_parameters.at("rope_theta"), "rope_parameters.rope_theta");
    }

    if (document.contains("rope_theta")) {
        return parse_positive_float(document.at("rope_theta"), "rope_theta");
    }

    return std::unexpected(invalid_field("rope_theta", "is missing"));
}

std::expected<void, ModelError>
validate_layer_types(const json::Document & document, std::size_t num_hidden_layers)
{
    if (!document.contains("layer_types")) {
        return {};
    }

    const auto & layer_types = document.at("layer_types");
    if (!layer_types.is_array()) [[unlikely]] {
        return std::unexpected(invalid_field("layer_types", "must be an array"));
    }
    if (layer_types.size() != num_hidden_layers) [[unlikely]] {
        return std::unexpected(invalid_field("layer_types", "length must equal num_hidden_layers"));
    }

    for (std::size_t index = 0; index < layer_types.size(); ++index) {
        const auto & layer_type = layer_types.at(index);
        if (!layer_type.is_string() || layer_type.get<std::string>() != "full_attention")
            [[unlikely]] {
            std::string field = "layer_types[" + std::to_string(index) + "]";
            return std::unexpected(
                invalid_field(field, "must be 'full_attention' in the current runtime")
            );
        }
    }

    return {};
}

std::expected<void, ModelError> validate_optional_runtime_features(const json::Document & document)
{
    if (document.contains("dtype")) {
        auto dtype = read_required_string(document, "dtype");
        if (!dtype) [[unlikely]] {
            return std::unexpected(std::move(dtype).error());
        }
        if (*dtype != "float32" && *dtype != "bfloat16") [[unlikely]] {
            return std::unexpected(
                invalid_field("dtype", "must be 'float32' or 'bfloat16' in the current runtime")
            );
        }
    }

    if (document.contains("use_sliding_window")) {
        auto use_sliding_window = read_required_boolean(document, "use_sliding_window");
        if (!use_sliding_window) [[unlikely]] {
            return std::unexpected(std::move(use_sliding_window).error());
        }
        if (*use_sliding_window) [[unlikely]] {
            return std::unexpected(
                invalid_field("use_sliding_window", "must be false in the current runtime")
            );
        }
    }

    if (document.contains("sliding_window") && !document.at("sliding_window").is_null())
        [[unlikely]] {
        return std::unexpected(
            invalid_field("sliding_window", "must be null in the current runtime")
        );
    }

    return {};
}

std::expected<json::Document, ModelError>
load_document(filesystem::Filesystem & filesystem, const std::filesystem::path & path)
{
    auto file_result = filesystem.open(path);
    if (!file_result) [[unlikely]] {
        return std::unexpected(std::move(file_result).error());
    }
    auto file = std::move(*file_result);

    auto file_size_result = file.size();
    if (!file_size_result) [[unlikely]] {
        return std::unexpected(std::move(file_size_result).error());
    }
    const auto file_size = *file_size_result;
    if (file_size > CONFIG_FILE_SIZE_LIMIT) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3 config file exceeds the 16 MiB size limit")
        );
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
    io::FileByteReader reader(file);
    auto read_result = reader.read_exact(bytes);
    if (!read_result) [[unlikely]] {
        return std::unexpected(std::move(read_result).error());
    }

    std::string text;
    if (!bytes.empty()) {
        text.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
    try {
        return json::Document::parse(text);
    } catch (const nlohmann::json::exception &) {
        return std::unexpected(invalid_configuration("Qwen3 config file is not valid JSON"));
    }
}

} // namespace

Qwen3Config::Qwen3Config(
    std::size_t vocab_size,
    std::size_t hidden_size,
    std::size_t intermediate_size,
    std::size_t num_hidden_layers,
    std::size_t num_attention_heads,
    std::size_t num_key_value_heads,
    std::size_t head_dim,
    std::size_t max_position_embeddings,
    float rms_norm_eps,
    float rope_theta,
    bool attention_bias,
    bool tie_word_embeddings
) noexcept
    : vocab_size_(vocab_size)
    , hidden_size_(hidden_size)
    , intermediate_size_(intermediate_size)
    , num_hidden_layers_(num_hidden_layers)
    , num_attention_heads_(num_attention_heads)
    , num_key_value_heads_(num_key_value_heads)
    , head_dim_(head_dim)
    , max_position_embeddings_(max_position_embeddings)
    , rms_norm_eps_(rms_norm_eps)
    , rope_theta_(rope_theta)
    , attention_bias_(attention_bias)
    , tie_word_embeddings_(tie_word_embeddings)
{}

std::expected<Qwen3Config, ModelError>
Qwen3Config::load(filesystem::Filesystem & filesystem, const std::filesystem::path & path)
{
    auto document = load_document(filesystem, path);
    if (!document) [[unlikely]] {
        return std::unexpected(std::move(document).error());
    }
    return from_json(*document);
}

std::expected<Qwen3Config, ModelError> Qwen3Config::from_json(const json::Document & document)
{
    try {
        if (!document.is_object()) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 config root must be a JSON object")
            );
        }

        auto model_type = read_required_string(document, "model_type");
        if (!model_type) [[unlikely]] {
            return std::unexpected(std::move(model_type).error());
        }
        if (*model_type != "qwen3") [[unlikely]] {
            return std::unexpected(invalid_field("model_type", "must be 'qwen3'"));
        }

        auto hidden_act = read_required_string(document, "hidden_act");
        if (!hidden_act) [[unlikely]] {
            return std::unexpected(std::move(hidden_act).error());
        }
        if (*hidden_act != "silu") [[unlikely]] {
            return std::unexpected(
                invalid_field("hidden_act", "must be 'silu' in the current runtime")
            );
        }

        auto vocab_size = read_positive_size(document, "vocab_size");
        if (!vocab_size) [[unlikely]] {
            return std::unexpected(std::move(vocab_size).error());
        }
        auto hidden_size = read_positive_size(document, "hidden_size");
        if (!hidden_size) [[unlikely]] {
            return std::unexpected(std::move(hidden_size).error());
        }
        auto intermediate_size = read_positive_size(document, "intermediate_size");
        if (!intermediate_size) [[unlikely]] {
            return std::unexpected(std::move(intermediate_size).error());
        }
        auto num_hidden_layers = read_positive_size(document, "num_hidden_layers");
        if (!num_hidden_layers) [[unlikely]] {
            return std::unexpected(std::move(num_hidden_layers).error());
        }
        auto num_attention_heads = read_positive_size(document, "num_attention_heads");
        if (!num_attention_heads) [[unlikely]] {
            return std::unexpected(std::move(num_attention_heads).error());
        }

        std::size_t num_key_value_heads = *num_attention_heads;
        if (document.contains("num_key_value_heads")) {
            auto parsed_num_key_value_heads = read_positive_size(document, "num_key_value_heads");
            if (!parsed_num_key_value_heads) [[unlikely]] {
                return std::unexpected(std::move(parsed_num_key_value_heads).error());
            }
            num_key_value_heads = *parsed_num_key_value_heads;
        }

        std::size_t head_dim = 0;
        if (document.contains("head_dim")) {
            auto parsed_head_dim = read_positive_size(document, "head_dim");
            if (!parsed_head_dim) [[unlikely]] {
                return std::unexpected(std::move(parsed_head_dim).error());
            }
            head_dim = *parsed_head_dim;
        } else {
            if (*hidden_size % *num_attention_heads != 0) [[unlikely]] {
                return std::unexpected(invalid_configuration(
                    "Qwen3 config hidden_size must be divisible by num_attention_heads "
                    "when head_dim is omitted"
                ));
            }
            head_dim = *hidden_size / *num_attention_heads;
        }

        auto max_position_embeddings = read_positive_size(document, "max_position_embeddings");
        if (!max_position_embeddings) [[unlikely]] {
            return std::unexpected(std::move(max_position_embeddings).error());
        }
        auto rms_norm_eps = read_positive_float(document, "rms_norm_eps");
        if (!rms_norm_eps) [[unlikely]] {
            return std::unexpected(std::move(rms_norm_eps).error());
        }
        auto rope_theta = read_rope_theta(document);
        if (!rope_theta) [[unlikely]] {
            return std::unexpected(std::move(rope_theta).error());
        }
        auto attention_bias = read_required_boolean(document, "attention_bias");
        if (!attention_bias) [[unlikely]] {
            return std::unexpected(std::move(attention_bias).error());
        }
        auto tie_word_embeddings = read_required_boolean(document, "tie_word_embeddings");
        if (!tie_word_embeddings) [[unlikely]] {
            return std::unexpected(std::move(tie_word_embeddings).error());
        }

        if (*num_attention_heads % num_key_value_heads != 0) [[unlikely]] {
            return std::unexpected(invalid_configuration(
                "Qwen3 config num_attention_heads must be divisible by "
                "num_key_value_heads"
            ));
        }
        if (head_dim % 2 != 0) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 config head_dim must be even for RoPE")
            );
        }
        if (head_dim > std::numeric_limits<std::size_t>::max() / *num_attention_heads ||
            head_dim > std::numeric_limits<std::size_t>::max() / num_key_value_heads) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 config attention projection dimension overflows")
            );
        }

        auto layer_types = validate_layer_types(document, *num_hidden_layers);
        if (!layer_types) [[unlikely]] {
            return std::unexpected(std::move(layer_types).error());
        }
        auto runtime_features = validate_optional_runtime_features(document);
        if (!runtime_features) [[unlikely]] {
            return std::unexpected(std::move(runtime_features).error());
        }

        return Qwen3Config(
            *vocab_size,
            *hidden_size,
            *intermediate_size,
            *num_hidden_layers,
            *num_attention_heads,
            num_key_value_heads,
            head_dim,
            *max_position_embeddings,
            *rms_norm_eps,
            *rope_theta,
            *attention_bias,
            *tie_word_embeddings
        );
    } catch (const nlohmann::json::exception &) {
        return std::unexpected(
            invalid_configuration("Qwen3 config contains an invalid JSON value")
        );
    }
}

std::size_t Qwen3Config::vocab_size() const noexcept
{
    return vocab_size_;
}

std::size_t Qwen3Config::hidden_size() const noexcept
{
    return hidden_size_;
}

std::size_t Qwen3Config::intermediate_size() const noexcept
{
    return intermediate_size_;
}

std::size_t Qwen3Config::num_hidden_layers() const noexcept
{
    return num_hidden_layers_;
}

std::size_t Qwen3Config::num_attention_heads() const noexcept
{
    return num_attention_heads_;
}

std::size_t Qwen3Config::num_key_value_heads() const noexcept
{
    return num_key_value_heads_;
}

std::size_t Qwen3Config::head_dim() const noexcept
{
    return head_dim_;
}

std::size_t Qwen3Config::max_position_embeddings() const noexcept
{
    return max_position_embeddings_;
}

float Qwen3Config::rms_norm_eps() const noexcept
{
    return rms_norm_eps_;
}

float Qwen3Config::rope_theta() const noexcept
{
    return rope_theta_;
}

bool Qwen3Config::attention_bias() const noexcept
{
    return attention_bias_;
}

bool Qwen3Config::tie_word_embeddings() const noexcept
{
    return tie_word_embeddings_;
}

} // namespace liteinfer::core::model::qwen3
