#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "core/common/data_type/data_type.hpp"
#include "core/tensor/tensor.hpp"

namespace
{

using liteinfer::core::common::data_type::BFloat16;
using liteinfer::core::common::data_type::DataType;
using liteinfer::core::common::data_type::DataTypeTraits;
using liteinfer::core::common::data_type::Float16;
using liteinfer::core::tensor::Shape;
using liteinfer::core::tensor::Tensor;

void test_float16()
{
    static_assert(sizeof(Float16) == 2);
    static_assert(std::is_trivially_copyable_v<Float16>);
    static_assert(DataTypeTraits<Float16>::value == DataType::Float16);

    assert(Float16(1.0F).bits() == 0x3c00);
    assert(Float16(-2.0F).bits() == 0xc000);
    assert(Float16(65504.0F).bits() == 0x7bff);

    assert(Float16::from_bits(0x3c00).to_float() == 1.0F);
    assert(Float16::from_bits(0x0001).to_float() == std::ldexp(1.0F, -24));
    assert(std::isinf(Float16::from_bits(0x7c00).to_float()));
    assert(std::isnan(Float16::from_bits(0x7e00).to_float()));

    const float f32_subnormal = std::bit_cast<float>(std::uint32_t {0x00000001});
    assert(Float16(f32_subnormal).bits() == 0x0000);
    assert(Float16(-f32_subnormal).bits() == 0x8000);

    // 最大 subnormal 与最小 normalized 之间的 round-to-even 边界。
    const float rounding_boundary = std::bit_cast<float>(std::uint32_t {0x387fe000});
    assert(Float16(rounding_boundary).bits() == 0x0400);
}

void test_bfloat16()
{
    static_assert(sizeof(BFloat16) == 2);
    static_assert(std::is_trivially_copyable_v<BFloat16>);
    static_assert(DataTypeTraits<BFloat16>::value == DataType::BFloat16);

    assert(BFloat16(1.0F).bits() == 0x3f80);
    assert(BFloat16(-2.0F).bits() == 0xc000);
    assert(BFloat16::from_bits(0x3f80).to_float() == 1.0F);
    assert(std::isinf(BFloat16::from_bits(0x7f80).to_float()));
    assert(std::isnan(BFloat16::from_bits(0x7fc1).to_float()));

    // BF16 最小 subnormal 在 float32 中的原始编码为 0x00010000。
    const float min_subnormal = std::bit_cast<float>(std::uint32_t {0x00010000});
    assert(BFloat16(min_subnormal).bits() == 0x0001);
    assert(BFloat16::from_bits(0x0001).to_float() == min_subnormal);

    // 1.0 与下一个 BF16 值之间的 round-to-nearest-even 边界。
    const float tie_to_even = std::bit_cast<float>(std::uint32_t {0x3f808000});
    const float tie_to_odd = std::bit_cast<float>(std::uint32_t {0x3f818000});
    assert(BFloat16(tie_to_even).bits() == 0x3f80);
    assert(BFloat16(tie_to_odd).bits() == 0x3f82);
}

void test_tensor_mapping()
{
    auto float16_result = Tensor::allocate(DataType::Float16, Shape {2});
    assert(float16_result.has_value());
    auto float16_values = float16_result->data_as<Float16>();
    assert(float16_values.has_value());
    (*float16_values)[0] = Float16(1.0F);
    assert((*float16_values)[0].to_float() == 1.0F);

    auto bfloat16_result = Tensor::allocate(DataType::BFloat16, Shape {2});
    assert(bfloat16_result.has_value());
    auto bfloat16_values = bfloat16_result->data_as<BFloat16>();
    assert(bfloat16_values.has_value());
    (*bfloat16_values)[0] = BFloat16(1.0F);
    assert((*bfloat16_values)[0].to_float() == 1.0F);
}

} // namespace

int main()
{
    test_float16();
    test_bfloat16();
    test_tensor_mapping();
}
