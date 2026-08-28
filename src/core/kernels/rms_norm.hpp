#pragma once

#include <cstddef>
#include <span>

namespace liteinfer::core::kernels
{

// RMS Norm
// input: 输入张量
// weight: 权重张量
// output: 输出张量
// width: 输入的宽度
// eps: 极小值，防止除以 0
void rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    std::size_t width,
    float eps
);

} // namespace liteinfer::core::kernels
