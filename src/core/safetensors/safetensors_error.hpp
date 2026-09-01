#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::safetensors
{

// Safetensors 错误码
enum class SafetensorsErrorCode : std::uint8_t
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

// Safetensors 错误类型
using SafetensorsError = common::Error;

} // namespace liteinfer::core::safetensors

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<safetensors::SafetensorsErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Safetensors;
};

} // namespace liteinfer::core::common
