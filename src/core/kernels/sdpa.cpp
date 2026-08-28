#include "core/kernels/sdpa.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace liteinfer::core::kernels
{

void sdpa_f32(
    std::span<const float> query,
    std::span<const float> key,
    std::span<const float> value,
    std::span<float> output,
    std::span<float> score_workspace,
    std::size_t head_dim,
    std::size_t query_position_offset)
{
    assert(head_dim > 0);
    assert(query.size() % head_dim == 0);
    assert(key.size() == value.size());
    assert(key.size() % head_dim == 0);
    assert(output.size() == query.size());

    const std::size_t query_length = query.size() / head_dim;
    const std::size_t key_length = key.size() / head_dim;
    assert(score_workspace.size() >= key_length);

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t query_index = 0; query_index < query_length; ++query_index) {
        const auto query_vector = query.subspan(query_index * head_dim, head_dim);

        const std::size_t query_position = query_position_offset + query_index;
        const std::size_t allowed_key_count = std::min(key_length, query_position + 1);

        float maximum_score = -std::numeric_limits<float>::infinity();

        for (std::size_t key_index = 0; key_index < allowed_key_count; ++key_index) {
            const auto key_vector = key.subspan(key_index * head_dim, head_dim);

            float score = 0.0F;
            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                score += query_vector[dimension] * key_vector[dimension];
            }

            score *= scale;
            score_workspace[key_index] = score;
            maximum_score = std::max(maximum_score, score);
        }

        float exponential_sum = 0.0F;
        for (std::size_t key_index = 0; key_index < allowed_key_count; ++key_index) {
            const float exponential = std::exp(score_workspace[key_index] - maximum_score);
            score_workspace[key_index] = exponential;
            exponential_sum += exponential;
        }

        auto output_vector = output.subspan(query_index * head_dim, head_dim);
        std::fill(output_vector.begin(), output_vector.end(), 0.0F);

        for (std::size_t key_index = 0; key_index < allowed_key_count; ++key_index) {
            const float probability = score_workspace[key_index] / exponential_sum;
            const auto value_vector = value.subspan(key_index * head_dim, head_dim);

            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                output_vector[dimension] += probability * value_vector[dimension];
            }
        }
    }
}

} // namespace liteinfer::core::kernels
