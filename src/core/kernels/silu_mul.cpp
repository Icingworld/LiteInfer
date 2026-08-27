#include "core/kernels/silu_mul.hpp"

#include <cassert>
#include <cmath>

namespace liteinfer::core::kernels
{

void silu_mul_f32(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output
)
{
    assert(gate.size() == up.size());
    assert(gate.size() == output.size());

    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float gate_value = gate[i];
        float sigmoid_value = 0.0F;
        if (gate_value >= 0.0F) {
            sigmoid_value = 1.0F / (1.0F + std::exp(-gate_value));
        } else {
            const float exp_value = std::exp(gate_value);
            sigmoid_value = exp_value / (1.0F + exp_value);
        }
        const float silu_value = gate_value * sigmoid_value;
        output[i] = silu_value * up[i];
    }
}

} // namespace liteinfer::core::kernels
