#pragma once

#include <span>

namespace liteinfer::core::kernels
{

// 矩阵加法
// lhs: 左矩阵
// rhs: 右矩阵
// output: 输出矩阵
void matadd_f32(std::span<const float> lhs, std::span<const float> rhs, std::span<float> output);

} // namespace liteinfer::core::kernels
