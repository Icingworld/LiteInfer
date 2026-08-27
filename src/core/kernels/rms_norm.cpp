#include "core/kernels/rms_norm.hpp"

#include <cassert>
#include <cmath>

namespace liteinfer::core::kernels
{

void rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    std::size_t width,
    float eps
)
{
    assert(input.size() == output.size());
    assert(input.size() % width == 0);
    assert(weight.size() == width);
    assert(width > 0);
    assert(eps > 0.0F);

    const std::size_t rows = input.size() / width;
    for (std::size_t row = 0; row < rows; ++row) {
        const float * row_input_start = input.data() + row * width;
        float * row_output_start = output.data() + row * width;

        float square_sum = 0.0F;

        // 计算平方和
        for (std::size_t i = 0; i < width; ++i) {
            square_sum += row_input_start[i] * row_input_start[i];
        }

        const float mean_square = square_sum / static_cast<float>(width);
        const float inverse_sqrt = 1.0F / std::sqrt(mean_square + eps);

        // 计算输出
        for (std::size_t i = 0; i < width; ++i) {
            row_output_start[i] = row_input_start[i] * inverse_sqrt * weight[i];
        }
    }
}

} // namespace liteinfer::core::kernels
