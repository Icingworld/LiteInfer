#include "core/common/data_type/bfloat16.hpp"

#include <bit>

namespace liteinfer::core::common::data_type
{

namespace
{

// 将 float32 转换为基于 IEEE 754 binary32 的非标准格式
[[nodiscard]]
std::uint16_t encode(float value) noexcept
{
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent = raw & 0x7f800000U;
    const std::uint32_t fraction = raw & 0x007fffffU;

    // Infinity / NaN
    if (exponent == 0x7f800000U) {
        if (fraction != 0) {
            // 保证结果为 quiet NaN
            return static_cast<std::uint16_t>((raw >> 16U) | 0x0040U);
        }

        return static_cast<std::uint16_t>(raw >> 16U);
    }

    // 丢弃低 16 位时进行 round-to-nearest-even
    const std::uint32_t rounding_bias = 0x7fffU + ((raw >> 16U) & 1U);

    return static_cast<std::uint16_t>((raw + rounding_bias) >> 16U);
}

// 将基于 IEEE 754 binary32 的非标准格式转换为 float32
[[nodiscard]]
float decode(std::uint16_t bits) noexcept
{
    const std::uint32_t raw = static_cast<std::uint32_t>(bits) << 16U;

    return std::bit_cast<float>(raw);
}

} // namespace

BFloat16::BFloat16() noexcept
    : bits_()
{}

BFloat16::BFloat16(float value) noexcept
    : bits_(encode(value))
{}

BFloat16 BFloat16::from_bits(std::uint16_t bits) noexcept
{
    BFloat16 result;
    result.bits_ = bits;
    return result;
}

std::uint16_t BFloat16::bits() const noexcept
{
    return bits_;
}

float BFloat16::to_float() const noexcept
{
    return decode(bits_);
}

BFloat16::operator float() const noexcept
{
    return to_float();
}

} // namespace liteinfer::core::common::data_type
