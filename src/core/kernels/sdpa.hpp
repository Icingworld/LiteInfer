#pragma once

#include <span>

namespace liteinfer::core::kernels
{

// 单头 Scaled Dot-Product Attention
// output = softmax(mask(query @ key^T / sqrt(head_dim))) @ value
// query: [query_length * head_dim]
// key/value: [key_length * head_dim]，key[j] 对应全局位置 j
// output: [query_length * head_dim]
// score_workspace: 至少 key_length，每个 query 行复用
// query_position_offset: 第一个 query 的全局位置（用于 causal mask）
void sdpa_f32(
    std::span<const float> query,
    std::span<const float> key,
    std::span<const float> value,
    std::span<float> output,
    std::span<float> score_workspace,
    std::size_t head_dim,
    std::size_t query_position_offset
);

} // namespace liteinfer::core::kernels
