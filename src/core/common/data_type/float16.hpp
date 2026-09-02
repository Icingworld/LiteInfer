#pragma once

#include <cstdint>

namespace liteinfer::core::common::data_type
{

// Float16 半精度浮点数
class Float16
{
public:
    // 初始化值为 +0
    Float16() noexcept;

    // float32 -> IEEE binary16，使用 round-to-nearest-even
    explicit Float16(float value) noexcept;

public:
    // 使用原始 binary16 值构造 Float16
    [[nodiscard]]
    static Float16 from_bits(std::uint16_t bits) noexcept;

    // 获取原始 binary16 值
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
