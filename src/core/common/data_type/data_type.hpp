#pragma once

#include <cstdint>

#include "core/common/data_type/bfloat16.hpp"
#include "core/common/data_type/float16.hpp"

namespace liteinfer::core::common::data_type
{

// 数据类型
enum class DataType : std::uint8_t
{
    Float16 = 0,
    BFloat16 = 1,
    Float32 = 2,
    // Float64 = 3,
    Int8 = 4,
    // Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    // Uint8 = 8,
    // Uint16 = 9,
    // Uint32 = 10,
    // Uint64 = 11,
    Bool = 12,
};

// C++ 类型到 Tensor 数据类型的映射
// 未特化的类型不属于当前 Tensor 的类型安全访问范围
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<Float16>
{
    static constexpr DataType value = DataType::Float16;
};

template <>
struct DataTypeTraits<BFloat16>
{
    static constexpr DataType value = DataType::BFloat16;
};

template <>
struct DataTypeTraits<float>
{
    static constexpr DataType value = DataType::Float32;
};

template <>
struct DataTypeTraits<std::int8_t>
{
    static constexpr DataType value = DataType::Int8;
};

template <>
struct DataTypeTraits<std::int32_t>
{
    static constexpr DataType value = DataType::Int32;
};

template <>
struct DataTypeTraits<std::int64_t>
{
    static constexpr DataType value = DataType::Int64;
};

template <>
struct DataTypeTraits<bool>
{
    static constexpr DataType value = DataType::Bool;
};

} // namespace liteinfer::core::common::data_type
