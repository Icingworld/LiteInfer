#include "core/kernels/rope.hpp"

#include <cassert>

namespace liteinfer::core::kernels
{

void rope_f32(
    std::span<const float> input,
    std::span<float> output,
    std::span<const float> cosine,
    std::span<const float> sine,
    std::size_t head_dim
)
{
    assert(input.size() == head_dim);
    assert(output.size() == head_dim);
    assert(cosine.size() == head_dim / 2);
    assert(sine.size() == head_dim / 2);
    assert(head_dim % 2 == 0);

    const std::size_t half_dim = head_dim / 2;

    for (std::size_t i = 0; i < half_dim; ++i) {
        const float first = input[i];
        const float second = input[i + half_dim];
        const float cos_value = cosine[i];
        const float sin_value = sine[i];

        output[i] = first * cos_value - second * sin_value;
        output[i + half_dim] = second * cos_value + first * sin_value;
    }
}

} // namespace liteinfer::core::kernels
