#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::safetensor
{

// Safetensor 错误码
enum class SafetensorErrorCode : std::uint8_t
{
    HeaderTooSmall = 0,
    HeaderTooLarge = 1,
    InvalidHeaderLength = 2,
    InvalidJson = 3,
    InvalidRoot = 4,
    InvalidMetadata = 5,
    InvalidTensorInfo = 6,
    InvalidDtype = 7,
    InvalidShape = 8,
    InvalidOffsets = 9,
    TensorSizeMismatch = 10,
    TensorNotFound = 11,
};

// Safetensor 错误类型
using SafetensorError = common::Error;

} // namespace liteinfer::core::safetensor

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<safetensor::SafetensorErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Safetensor;
};

} // namespace liteinfer::core::common
