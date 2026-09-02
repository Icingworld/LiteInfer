#include "core/common/data_type/float16.hpp"

#include <bit>

namespace liteinfer::core::common::data_type
{

namespace
{

// 将 32 位整数右移，使用 round-to-nearest-even
[[nodiscard]]
std::uint32_t round_shift_to_even(std::uint32_t value, int shift) noexcept
{
    const std::uint32_t truncated = value >> shift;
    const std::uint32_t remainder = value & ((std::uint32_t {1} << shift) - 1U);
    const std::uint32_t halfway = std::uint32_t {1} << (shift - 1);

    if (remainder > halfway || (remainder == halfway && (truncated & 1U) != 0)) {
        return truncated + 1U;
    }

    return truncated;
}

// 将 float32 转换为 IEEE binary16，使用 round-to-nearest-even
[[nodiscard]]
std::uint16_t encode(float value) noexcept
{
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((raw >> 16U) & 0x8000U);
    const int exponent = static_cast<int>((raw >> 23U) & 0xffU);
    const std::uint32_t fraction = raw & 0x007fffffU;

    // Infinity / NaN
    if (exponent == 0xff) {
        if (fraction == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }

        std::uint16_t payload = static_cast<std::uint16_t>(fraction >> 13U);

        // 确保结果仍然是 NaN
        if (payload == 0) {
            payload = 1;
        }

        return static_cast<std::uint16_t>(sign | 0x7c00U | payload);
    }

    int half_exponent = exponent - 127 + 15;

    // 溢出为 Infinity
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    // 结果为 subnormal 或零
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }

        const std::uint32_t mantissa = fraction | 0x00800000U;
        const int shift = 14 - half_exponent;
        const std::uint32_t rounded = round_shift_to_even(mantissa, shift);

        return static_cast<std::uint16_t>(sign | rounded);
    }

    // 普通 normalized 数
    std::uint32_t half_fraction = fraction >> 13U;
    const std::uint32_t discarded = fraction & 0x1fffU;

    if (discarded > 0x1000U || (discarded == 0x1000U && (half_fraction & 1U) != 0)) {
        ++half_fraction;

        // 尾数进位导致指数增加
        if (half_fraction == 0x400U) {
            half_fraction = 0;
            ++half_exponent;

            if (half_exponent >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00U);
            }
        }
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(half_exponent) << 10U) |
        half_fraction
    );
}

// 将 IEEE binary16 转换为 float32
[[nodiscard]]
float decode(std::uint16_t bits) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t fraction = bits & 0x03ffU;

    std::uint32_t raw;

    if (exponent == 0) {
        if (fraction == 0) {
            raw = sign;
        } else {
            // 处理 half subnormal
            int normalized_exponent = -14;
            std::uint32_t normalized_fraction = fraction;

            while ((normalized_fraction & 0x400U) == 0) {
                normalized_fraction <<= 1U;
                --normalized_exponent;
            }

            normalized_fraction &= 0x03ffU;

            raw =
                sign |
                (static_cast<std::uint32_t>(
                        normalized_exponent + 127
                    ) << 23U) |
                (normalized_fraction << 13U);
        }
    } else if (exponent == 0x1f) {
        // Infinity / NaN
        raw = sign | 0x7f800000U | (fraction << 13U);
    } else {
        // normalized number
        const std::uint32_t float_exponent = exponent - 15U + 127U;

        raw =
            sign |
            (float_exponent << 23U) |
            (fraction << 13U);
    }

    return std::bit_cast<float>(raw);
}

} // namespace

Float16::Float16() noexcept
    : bits_()
{}

Float16::Float16(float value) noexcept
    : bits_(encode(value))
{}

Float16 Float16::from_bits(std::uint16_t bits) noexcept
{
    Float16 result;
    result.bits_ = bits;
    return result;
}

std::uint16_t Float16::bits() const noexcept
{
    return bits_;
}

float Float16::to_float() const noexcept
{
    return decode(bits_);
}

Float16::operator float() const noexcept
{
    return to_float();
}

} // namespace liteinfer::core::common::data_type
