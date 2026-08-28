#pragma once

#include <cstddef>
#include <span>

namespace liteinfer::core::kernels
{

// 计算旋转位置编码
// 使用 LLaMA/Qwen 风格的 half-split RoPE 实现
// input: 输入张量
// output: 输出张量
// cosine: 余弦值
// sine: 正弦值
// head_dim: 头维度
void rope_f32(
    std::span<const float> input,
    std::span<float> output,
    std::span<const float> cosine,
    std::span<const float> sine,
    std::size_t head_dim
);

} // namespace liteinfer::core::kernels
