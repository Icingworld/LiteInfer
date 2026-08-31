#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::io
{

// IO 错误码
enum class IoErrorCode : std::uint8_t
{
    UnexpectedEof = 0,
    InvalidData = 1,
    ValueTooLarge = 2,
};

// IO 错误类型
using IoError = common::Error;

} // namespace liteinfer::core::io

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<io::IoErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Io;
};

} // namespace liteinfer::core::common
