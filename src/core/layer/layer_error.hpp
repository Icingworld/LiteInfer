#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::layer
{

// Layer 错误码
enum class LayerErrorCode : std::uint8_t
{
    InvalidWeight = 0,
    InvalidBias = 1,
    InvalidInput = 2,
    UnsupportedDataType = 3,
};

// Layer 错误类型
using LayerError = common::Error;

} // namespace liteinfer::core::layer

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<layer::LayerErrorCode>
{
    static constexpr common::ErrorCategory category = common::ErrorCategory::Layer;
};

} // namespace liteinfer::core::common
