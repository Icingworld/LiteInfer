#pragma once

#include <span>

namespace liteinfer::core::kernels
{

// SwiGLU 门控乘法
// gate: gate 分支线性层输出
// up: up 分支线性层输出
// output: 逐元素乘积结果
void silu_mul_f32(std::span<const float> gate, std::span<const float> up, std::span<float> output);

} // namespace liteinfer::core::kernels
