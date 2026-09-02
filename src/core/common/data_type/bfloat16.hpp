#pragma once

#include <cstdint>

namespace liteinfer::core::common::data_type
{

// BFloat16：1 位符号位、8 位指数位、7 位分数位
class BFloat16
{
public:
    // 初始化值为 +0
    BFloat16() noexcept;

    // float32 -> 基于 IEEE 754 binary32 的非标准格式
    explicit BFloat16(float value) noexcept;

public:
    // 使用原始 bfloat16 编码构造 BFloat16
    [[nodiscard]]
    static BFloat16 from_bits(std::uint16_t bits) noexcept;

    // 获取原始 bfloat16 编码
    [[nodiscard]]
    std::uint16_t bits() const noexcept;

    // 获取 float32 值，此转换是精确的
    [[nodiscard]]
    float to_float() const noexcept;

    // 转换为 float32
    [[nodiscard]]
    explicit operator float() const noexcept;

private:
    std::uint16_t bits_;
};

} // namespace liteinfer::core::common::data_type
