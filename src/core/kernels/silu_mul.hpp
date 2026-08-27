#pragma once

#include <span>

namespace liteinfer::core::kernels
{

// 门控激活函数乘法
// gate: 门控激活函数
// up: 上层激活函数
// output: 输出
void silu_mul_f32(std::span<const float> gate, std::span<const float> up, std::span<float> output);

} // namespace liteinfer::core::kernels
