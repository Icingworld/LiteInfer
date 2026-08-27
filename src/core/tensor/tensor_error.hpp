#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::tensor
{

// Tensor 错误码
enum class TensorErrorCode : std::uint8_t
{
    IndexOutOfRange = 0,
    Overflow = 1,
    InvalidDataType = 2,
    InvalidByteSize = 3,
    DataTypeMismatch = 4,
    NonContiguous = 5,
    InvalidDataAlignment = 6,
};

// Tensor 错误
using TensorError = common::Error;

} // namespace liteinfer::core::tensor

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<tensor::TensorErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Tensor;
};

} // namespace liteinfer::core::common
