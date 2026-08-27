#include "core/kernels/matmul.hpp"

#include <cassert>

namespace liteinfer::core::kernels
{

void matmul_f32(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output,
    std::size_t m,
    std::size_t k,
    std::size_t n
)
{
    assert(lhs.size() == m * k);
    assert(rhs.size() == k * n);
    assert(output.size() == m * n);
    assert(m > 0);
    assert(k > 0);
    assert(n > 0);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0F;
            
            for (std::size_t inner = 0; inner < k; ++inner) {
                const float lhs_value = lhs[row * k + inner];
                const float rhs_value = rhs[inner * n + col];

                sum += lhs_value * rhs_value;
            }

            output[row * n + col] = sum;
        }
    }
}

} // namespace liteinfer::core::kernels
