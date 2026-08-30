#include "core/model/qwen3/qwen3_attention.hpp"

#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "core/kernels/rope.hpp"
#include "core/kernels/sdpa.hpp"

namespace liteinfer::core::model::qwen3
{

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
        num_attention_heads % num_key_value_heads != 0) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention head counts must be non-zero and query heads must divide by KV heads"
        ));
    }

    const std::size_t head_dim = q_norm.normalized_size();
    if (head_dim == 0 || head_dim % 2 != 0 || k_norm.normalized_size() != head_dim) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention Q/K norms must use the same non-zero even head dimension"
        ));
    }

    if (q_norm.eps() != k_norm.eps()) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention Q/K norms must use the same epsilon"
        ));
    }

    if (!std::isfinite(rope_theta) || rope_theta <= 0.0F) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention RoPE theta must be finite and greater than zero"
        ));
    }

    if (head_dim > std::numeric_limits<std::size_t>::max() / num_attention_heads ||
        head_dim > std::numeric_limits<std::size_t>::max() / num_key_value_heads) {
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
        o_proj.out_features() != hidden_size) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidConfiguration,
            "Qwen3 Attention projection dimensions do not match the head configuration"
        ));
    }

    const bool attention_bias = q_proj.has_bias();
    if (k_proj.has_bias() != attention_bias || v_proj.has_bias() != attention_bias ||
        o_proj.has_bias() != attention_bias) {
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
    if (input.rank() < 2) {
        return std::unexpected(ModelError(
            ModelErrorCode::InvalidInput,
            "Qwen3 Attention input must have shape [..., sequence_length, hidden_size]"
        ));
    }

    auto query = q_proj_.forward(input);
    if (!query) {
        return std::unexpected(std::move(query).error());
    }

    if (input.empty()) {
        return o_proj_.forward(*query);
    }

    auto key = k_proj_.forward(input);
    if (!key) {
        return std::unexpected(std::move(key).error());
    }

    auto value = v_proj_.forward(input);
    if (!value) {
        return std::unexpected(std::move(value).error());
    }

    auto query_rows = tensor::Tensor::from_bytes(
        tensor::DataType::Float32,
        tensor::Shape {query->numel() / head_dim_, head_dim_},
        query->data()
    );
    if (!query_rows) {
        return std::unexpected(std::move(query_rows).error());
    }

    auto key_rows = tensor::Tensor::from_bytes(
        tensor::DataType::Float32,
        tensor::Shape {key->numel() / head_dim_, head_dim_},
        key->data()
    );
    if (!key_rows) {
        return std::unexpected(std::move(key_rows).error());
    }

    auto normalized_query = q_norm_.forward(*query_rows);
    if (!normalized_query) {
        return std::unexpected(std::move(normalized_query).error());
    }

    auto normalized_key = k_norm_.forward(*key_rows);
    if (!normalized_key) {
        return std::unexpected(std::move(normalized_key).error());
    }

    auto rotated_query =
        tensor::Tensor::allocate(tensor::DataType::Float32, normalized_query->shape());
    if (!rotated_query) {
        return std::unexpected(std::move(rotated_query).error());
    }

    auto rotated_key = tensor::Tensor::allocate(tensor::DataType::Float32, normalized_key->shape());
    if (!rotated_key) {
        return std::unexpected(std::move(rotated_key).error());
    }

    auto normalized_query_values = normalized_query->data_as<float>();
    if (!normalized_query_values) {
        return std::unexpected(std::move(normalized_query_values).error());
    }

    auto normalized_key_values = normalized_key->data_as<float>();
    if (!normalized_key_values) {
        return std::unexpected(std::move(normalized_key_values).error());
    }

    auto rotated_query_values = rotated_query->data_as<float>();
    if (!rotated_query_values) {
        return std::unexpected(std::move(rotated_query_values).error());
    }

    auto rotated_key_values = rotated_key->data_as<float>();
    if (!rotated_key_values) {
        return std::unexpected(std::move(rotated_key_values).error());
    }

    const auto input_shape = input.shape().values();
    const std::size_t sequence_length = input_shape[input_shape.size() - 2];
    const std::size_t query_width = num_attention_heads_ * head_dim_;
    const std::size_t key_value_width = num_key_value_heads_ * head_dim_;
    const std::size_t token_count = query->numel() / query_width;
    const std::size_t batch_count = token_count / sequence_length;
    const std::size_t half_dim = head_dim_ / 2;

    std::vector<float> inverse_frequencies(half_dim);
    for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
        const float exponent = 2.0F * static_cast<float>(dimension) / static_cast<float>(head_dim_);
        inverse_frequencies[dimension] = 1.0F / std::pow(rope_theta_, exponent);
    }

    std::vector<float> cosine(sequence_length * half_dim);
    std::vector<float> sine(sequence_length * half_dim);
    for (std::size_t position = 0; position < sequence_length; ++position) {
        for (std::size_t dimension = 0; dimension < half_dim; ++dimension) {
            const float angle = static_cast<float>(position) * inverse_frequencies[dimension];
            cosine[position * half_dim + dimension] = std::cos(angle);
            sine[position * half_dim + dimension] = std::sin(angle);
        }
    }

    for (std::size_t token = 0; token < token_count; ++token) {
        const std::size_t position = token % sequence_length;
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

    auto attention_output = tensor::Tensor::allocate(tensor::DataType::Float32, query->shape());
    if (!attention_output) {
        return std::unexpected(std::move(attention_output).error());
    }

    auto value_values = value->data_as<float>();
    if (!value_values) {
        return std::unexpected(std::move(value_values).error());
    }

    auto attention_output_values = attention_output->data_as<float>();
    if (!attention_output_values) {
        return std::unexpected(std::move(attention_output_values).error());
    }

    std::vector<float> query_head(sequence_length * head_dim_);
    std::vector<float> key_head(sequence_length * head_dim_);
    std::vector<float> value_head(sequence_length * head_dim_);
    std::vector<float> head_output(sequence_length * head_dim_);
    std::vector<float> score_workspace(sequence_length);
    const std::size_t query_heads_per_key_value_head = num_attention_heads_ / num_key_value_heads_;

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
                    key_head[position * head_dim_ + dimension] = (*rotated_key_values)
                        [(token * num_key_value_heads_ + key_value_head_index) * head_dim_ +
                         dimension];
                    value_head[position * head_dim_ + dimension] = (*value_values)
                        [token * key_value_width + key_value_head_index * head_dim_ + dimension];
                }
            }

            kernels::sdpa_f32(
                query_head,
                key_head,
                value_head,
                head_output,
                score_workspace,
                head_dim_,
                0
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
