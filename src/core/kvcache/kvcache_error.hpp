#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::kvcache
{

// KV cache 错误码
enum class KVCacheErrorCode : std::uint8_t
{
    InvalidConfiguration = 0,
    InvalidOperation = 1,
    CapacityExceeded = 2,
    InvalidLayer = 3,
    ShapeMismatch = 4,
    DataTypeMismatch = 5,
};

// KV cache 错误类型
using KVCacheError = common::Error;

} // namespace liteinfer::core::kvcache

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<kvcache::KVCacheErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::KVCache;
};

} // namespace liteinfer::core::common
