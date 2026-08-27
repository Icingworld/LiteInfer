#include "core/kernels/matadd.hpp"

#include <cassert>

namespace liteinfer::core::kernels
{

void matadd_f32(std::span<const float> lhs, std::span<const float> rhs, std::span<float> output)
{
    assert(lhs.size() == rhs.size());
    assert(lhs.size() == output.size());

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        output[i] = lhs[i] + rhs[i];
    }
}

} // namespace liteinfer::core::kernels
