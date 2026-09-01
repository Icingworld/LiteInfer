#include "core/model/qwen3/qwen3_attention.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "core/kernels/rope.hpp"
#include "core/kernels/sdpa.hpp"

namespace liteinfer::core::model::qwen3
{

namespace
{

ModelError cache_error(const kvcache::KVCacheError & error)
{
    return ModelError(ModelErrorCode::InvalidInput, error.message());
}

std::expected<std::span<const float>, ModelError> cache_float_storage(const tensor::Tensor & value)
{
    if (value.data_type() != tensor::DataType::Float32) [[unlikely]] {
        return std::unexpected(
            ModelError(ModelErrorCode::InvalidInput, "Qwen3 KV cache must use Float32")
        );
    }

    const auto bytes = value.data();
    if (bytes.size() % sizeof(float) != 0) [[unlikely]] {
        return std::unexpected(
            ModelError(ModelErrorCode::InvalidInput, "Qwen3 KV cache byte span is invalid")
        );
    }

    if (bytes.empty()) {
        return std::span<const float>();
    }

    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    if (address % alignof(float) != 0) [[unlikely]] {
        return std::unexpected(
            ModelError(ModelErrorCode::InvalidInput, "Qwen3 KV cache data alignment is invalid")
        );
    }

    return std::span<const float>(
        reinterpret_cast<const float *>(bytes.data()),
        bytes.size() / sizeof(float)
    );
}

} // namespace

Qwen3Attention::Qwen3Attention(
    layer::Linear q_proj,
    layer::Linear k_proj,
    layer::Linear v_proj,
    layer::Linear o_proj,
    layer::RMSNorm q_norm,
    layer::RMSNorm k_norm,
    std::size_t num_attention_heads,
    std::size_t num_key_value_heads,
    std::size_t head_dim,
    float rope_theta
)
    : q_proj_(std::move(q_proj))
    , k_proj_(std::move(k_proj))
    , v_proj_(std::move(v_proj))
    , o_proj_(std::move(o_proj))
    , q_norm_(std::move(q_norm))
    , k_norm_(std::move(k_norm))
    , num_attention_heads_(num_attention_heads)
    , num_key_value_heads_(num_key_value_heads)
    , head_dim_(head_dim)
    , rope_theta_(rope_theta)
{}

std::expected<Qwen3Attention, ModelError> Qwen3Attention::create(
    layer::Linear q_proj,
    layer::Linear k_proj,
    layer::Linear v_proj,
    layer::Linear o_proj,
    layer::RMSNorm q_norm,
    layer::RMSNorm k_norm,
    std::size_t num_attention_heads,
    std::size_t num_key_value_heads,
    float rope_theta
)
{
    if (num_attention_heads == 0 || num_key_value_heads == 0 ||
        num_attention_heads % num_key_value_heads != 0) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention head counts must be non-zero and query heads must divide by KV heads"
        ));
    }

    const std::size_t head_dim = q_norm.normalized_size();
    if (head_dim == 0 || head_dim % 2 != 0 || k_norm.normalized_size() != head_dim) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention Q/K norms must use the same non-zero even head dimension"
        ));
    }

    if (q_norm.eps() != k_norm.eps()) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention Q/K norms must use the same epsilon"
        ));
    }

    if (!std::isfinite(rope_theta) || rope_theta <= 0.0F) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention RoPE theta must be finite and greater than zero"
        ));
    }

    if (head_dim > std::numeric_limits<std::size_t>::max() / num_attention_heads ||
        head_dim > std::numeric_limits<std::size_t>::max() / num_key_value_heads) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention projection dimension overflow"
        ));
    }

    const std::size_t query_width = num_attention_heads * head_dim;
    const std::size_t key_value_width = num_key_value_heads * head_dim;
    const std::size_t hidden_size = q_proj.in_features();

    if (q_proj.out_features() != query_width || k_proj.in_features() != hidden_size ||
        k_proj.out_features() != key_value_width || v_proj.in_features() != hidden_size ||
        v_proj.out_features() != key_value_width || o_proj.in_features() != query_width ||
        o_proj.out_features() != hidden_size) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention projection dimensions do not match the head configuration"
        ));
    }

    const bool attention_bias = q_proj.has_bias();
    if (k_proj.has_bias() != attention_bias || v_proj.has_bias() != attention_bias ||
        o_proj.has_bias() != attention_bias) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention projections must use a consistent bias configuration"
        ));
    }

    return Qwen3Attention(
        std::move(q_proj),
        std::move(k_proj),
        std::move(v_proj),
        std::move(o_proj),
        std::move(q_norm),
        std::move(k_norm),
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rope_theta
    );
}

std::expected<tensor::Tensor, ModelError> Qwen3Attention::forward(
    const tensor::Tensor & input
) const
{
    return forward_impl(input, nullptr, 0, nullptr);
}

std::expected<tensor::Tensor, ModelError> Qwen3Attention::forward(
    const tensor::Tensor & input,
    kvcache::KVCache & cache,
    std::size_t layer_index,
    const kvcache::KVCacheRegion & region
) const
{
    return forward_impl(input, &cache, layer_index, &region);
}

std::expected<tensor::Tensor, ModelError> Qwen3Attention::forward_impl(
    const tensor::Tensor & input,
    kvcache::KVCache * cache,
    std::size_t layer_index,
    const kvcache::KVCacheRegion * region
) const
{
    const bool use_cache = cache != nullptr;
    if (use_cache != (region != nullptr)) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention cache and cache region must be provided together"
        ));
    }

    if (input.rank() < 2) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention input must have shape [..., sequence_length, hidden_size]"
        ));
    }

    auto query = q_proj_.forward(input);
    if (!query) [[unlikely]] {
        return std::unexpected(std::move(query).error());
    }

    if (input.empty()) [[unlikely]] {
        if (use_cache) [[unlikely]] {
            return std::unexpected(ModelError(
                ModelErrorCode::InvalidInput,
                "Qwen3 Attention cached input sequence must not be empty"
            ));
        }
        return o_proj_.forward(*query);
    }

    auto key = k_proj_.forward(input);
    if (!key) [[unlikely]] {
        return std::unexpected(std::move(key).error());
    }

    auto value = v_proj_.forward(input);
    if (!value) [[unlikely]] {
        return std::unexpected(std::move(value).error());
    }

    auto query_rows = tensor::Tensor::from_bytes(
        tensor::DataType::Float32,
        tensor::Shape {query->numel() / head_dim_, head_dim_},
        query->data()
    );
    if (!query_rows) [[unlikely]] {
        return std::unexpected(std::move(query_rows).error());
    }

    auto key_rows = tensor::Tensor::from_bytes(
        tensor::DataType::Float32,
        tensor::Shape {key->numel() / head_dim_, head_dim_},
        key->data()
    );
    if (!key_rows) [[unlikely]] {
        return std::unexpected(std::move(key_rows).error());
    }

    auto normalized_query = q_norm_.forward(*query_rows);
    if (!normalized_query) [[unlikely]] {
        return std::unexpected(std::move(normalized_query).error());
    }

    auto normalized_key = k_norm_.forward(*key_rows);
    if (!normalized_key) [[unlikely]] {
        return std::unexpected(std::move(normalized_key).error());
    }

    auto rotated_query =
        tensor::Tensor::allocate(tensor::DataType::Float32, normalized_query->shape());
    if (!rotated_query) [[unlikely]] {
        return std::unexpected(std::move(rotated_query).error());
    }

    auto rotated_key = tensor::Tensor::allocate(tensor::DataType::Float32, normalized_key->shape());
    if (!rotated_key) [[unlikely]] {
        return std::unexpected(std::move(rotated_key).error());
    }

    auto normalized_query_values = normalized_query->data_as<float>();
    if (!normalized_query_values) [[unlikely]] {
        return std::unexpected(std::move(normalized_query_values).error());
    }

    auto normalized_key_values = normalized_key->data_as<float>();
    if (!normalized_key_values) [[unlikely]] {
        return std::unexpected(std::move(normalized_key_values).error());
    }

    auto rotated_query_values = rotated_query->data_as<float>();
    if (!rotated_query_values) [[unlikely]] {
        return std::unexpected(std::move(rotated_query_values).error());
    }

    auto rotated_key_values = rotated_key->data_as<float>();
    if (!rotated_key_values) [[unlikely]] {
        return std::unexpected(std::move(rotated_key_values).error());
    }

    const auto input_shape = input.shape().values();
    const std::size_t sequence_length = input_shape[input_shape.size() - 2];
    const std::size_t query_width = num_attention_heads_ * head_dim_;
    const std::size_t key_value_width = num_key_value_heads_ * head_dim_;
    const std::size_t token_count = query->numel() / query_width;
    if (sequence_length == 0 || token_count % sequence_length != 0) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention input has an inconsistent sequence dimension"
        ));
    }

    const std::size_t batch_count = token_count / sequence_length;
    const std::size_t position_offset = use_cache ? region->start : 0;
    if (use_cache) {
        if (input.rank() != 2 && input.rank() != 3) [[unlikely]] {
            return std::unexpected(ModelError(
                ModelErrorCode::InvalidInput,
                "Qwen3 Attention cached input must be 2D or 3D"
            ));
        }
        if (batch_count != 1 || cache->batch_size() != 1 || region->count != sequence_length ||
            region->start != cache->length() || region->end > cache->capacity()) [[unlikely]] {
            return std::unexpected(ModelError(
                ModelErrorCode::InvalidInput,
                "Qwen3 Attention cache region does not match the input sequence"
            ));
        }
    }

    const std::size_t half_dim = head_dim_ / 2;
    if (sequence_length > std::numeric_limits<std::size_t>::max() / half_dim) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention RoPE workspace size overflows size_t"
        ));
    }

    std::vector<float> inverse_frequencies(half_dim);
    for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
        const float exponent = 2.0F * static_cast<float>(dimension) / static_cast<float>(head_dim_);
        inverse_frequencies[dimension] = 1.0F / std::pow(rope_theta_, exponent);
    }

    std::vector<float> cosine(sequence_length * half_dim);
    std::vector<float> sine(sequence_length * half_dim);
    for (std::size_t position = 0; position < sequence_length; ++position) {
        const std::size_t absolute_position = position_offset + position;
        for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
            const float angle =
                static_cast<float>(absolute_position) * inverse_frequencies[dimension];
            cosine[position * half_dim + dimension] = std::cos(angle);
            sine[position * half_dim + dimension] = std::sin(angle);
        }
    }

    for (std::size_t token = 0; token < token_count; ++token) {
        const std::size_t position = use_cache ? token : token % sequence_length;
        const auto position_cosine =
            std::span<const float>(cosine).subspan(position * half_dim, half_dim);
        const auto position_sine =
            std::span<const float>(sine).subspan(position * half_dim, half_dim);

        for (std::size_t head = 0; head < num_attention_heads_; ++head) {
            const std::size_t row = token * num_attention_heads_ + head;
            kernels::rope_f32(
                normalized_query_values->subspan(row * head_dim_, head_dim_),
                rotated_query_values->subspan(row * head_dim_, head_dim_),
                position_cosine,
                position_sine,
                head_dim_
            );
        }

        for (std::size_t head = 0; head < num_key_value_heads_; ++head) {
            const std::size_t row = token * num_key_value_heads_ + head;
            kernels::rope_f32(
                normalized_key_values->subspan(row * head_dim_, head_dim_),
                rotated_key_values->subspan(row * head_dim_, head_dim_),
                position_cosine,
                position_sine,
                head_dim_
            );
        }
    }

    if (use_cache) {
        auto value_rows = tensor::Tensor::from_bytes(
            tensor::DataType::Float32,
            tensor::Shape {value->numel() / head_dim_, head_dim_},
            value->data()
        );
        if (!value_rows) [[unlikely]] {
            return std::unexpected(std::move(value_rows).error());
        }

        auto cache_write =
            cache->write_token_major(layer_index, *region, *rotated_key, *value_rows);
        if (!cache_write) [[unlikely]] {
            return std::unexpected(cache_error(cache_write.error()));
        }
    }

    auto attention_output = tensor::Tensor::allocate(tensor::DataType::Float32, query->shape());
    if (!attention_output) [[unlikely]] {
        return std::unexpected(std::move(attention_output).error());
    }

    auto value_values = value->data_as<float>();
    if (!value_values) [[unlikely]] {
        return std::unexpected(std::move(value_values).error());
    }

    auto attention_output_values = attention_output->data_as<float>();
    if (!attention_output_values) [[unlikely]] {
        return std::unexpected(std::move(attention_output_values).error());
    }

    const std::size_t key_length = use_cache ? region->end : sequence_length;
    if (key_length > std::numeric_limits<std::size_t>::max() / head_dim_) [[unlikely]] {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention workspace size overflows size_t"
        ));
    }

    std::vector<float> query_head(sequence_length * head_dim_);
    std::vector<float> head_output(sequence_length * head_dim_);
    std::vector<float> score_workspace(key_length);
    std::vector<float> key_head;
    std::vector<float> value_head;
    if (!use_cache) {
        key_head.resize(key_length * head_dim_);
        value_head.resize(key_length * head_dim_);
    }
    const std::size_t query_heads_per_key_value_head = num_attention_heads_ / num_key_value_heads_;

    std::span<const float> cache_key_values;
    std::span<const float> cache_value_values;
    std::size_t cache_head_stride = 0;
    if (use_cache) {
        auto cache_view = cache->view(layer_index, region->end);
        if (!cache_view) [[unlikely]] {
            return std::unexpected(cache_error(cache_view.error()));
        }

        auto key_values = cache_float_storage(cache_view->key);
        if (!key_values) [[unlikely]] {
            return std::unexpected(std::move(key_values).error());
        }
        auto cached_value_values = cache_float_storage(cache_view->value);
        if (!cached_value_values) [[unlikely]] {
            return std::unexpected(std::move(cached_value_values).error());
        }

        const auto key_strides = cache_view->key.strides().values();
        const auto value_strides = cache_view->value.strides().values();
        if (key_strides.size() != 4 || value_strides.size() != 4 || key_strides[3] != 1 ||
            value_strides[3] != 1 || key_strides[2] != head_dim_ || value_strides[2] != head_dim_ ||
            key_strides[1] != value_strides[1]) [[unlikely]] {
            return std::unexpected(ModelError(
                ModelErrorCode::InvalidInput,
                "Qwen3 KV cache layout is not compatible with attention"
            ));
        }

        cache_key_values = *key_values;
        cache_value_values = *cached_value_values;
        cache_head_stride = key_strides[1];
    }

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t query_head_index = 0; query_head_index < num_attention_heads_;
             ++query_head_index) {
            const std::size_t key_value_head_index =
                query_head_index / query_heads_per_key_value_head;

            for (std::size_t position = 0; position < sequence_length; ++position) {
                const std::size_t token = batch * sequence_length + position;
                for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                    query_head[position * head_dim_ + dimension] = (*rotated_query_values)
                        [(token * num_attention_heads_ + query_head_index) * head_dim_ + dimension];
                }
            }

            if (!use_cache) {
                for (std::size_t position = 0; position < sequence_length; ++position) {
                    const std::size_t token = batch * sequence_length + position;
                    for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                        key_head[position * head_dim_ + dimension] = (*rotated_key_values)
                            [(token * num_key_value_heads_ + key_value_head_index) * head_dim_ +
                             dimension];
                        value_head[position * head_dim_ + dimension] = (*value_values)
                            [token * key_value_width + key_value_head_index * head_dim_ +
                             dimension];
                    }
                }
            }

            const std::span<const float> attention_keys = [&]() {
                if (!use_cache) {
                    return std::span<const float>(key_head);
                }

                const std::size_t head_offset = key_value_head_index * cache_head_stride;
                return cache_key_values.subspan(head_offset, key_length * head_dim_);
            }();
            const std::span<const float> attention_values = [&]() {
                if (!use_cache) {
                    return std::span<const float>(value_head);
                }

                const std::size_t head_offset = key_value_head_index * cache_head_stride;
                return cache_value_values.subspan(head_offset, key_length * head_dim_);
            }();

            kernels::sdpa_f32(
                query_head,
                attention_keys,
                attention_values,
                head_output,
                score_workspace,
                head_dim_,
                position_offset
            );

            for (std::size_t position = 0; position < sequence_length; ++position) {
                const std::size_t token = batch * sequence_length + position;
                for (std::size_t dimension = 0; dimension < head_dim_; ++dimension) {
                    (*attention_output_values)
                        [token * query_width + query_head_index * head_dim_ + dimension] =
                            head_output[position * head_dim_ + dimension];
                }
            }
        }
    }

    return o_proj_.forward(*attention_output);
}

std::size_t Qwen3Attention::hidden_size() const noexcept
{
    return q_proj_.in_features();
}

std::size_t Qwen3Attention::num_attention_heads() const noexcept
{
    return num_attention_heads_;
}

std::size_t Qwen3Attention::num_key_value_heads() const noexcept
{
    return num_key_value_heads_;
}

std::size_t Qwen3Attention::head_dim() const noexcept
{
    return head_dim_;
}

float Qwen3Attention::rope_theta() const noexcept
{
    return rope_theta_;
}

} // namespace liteinfer::core::model::qwen3
