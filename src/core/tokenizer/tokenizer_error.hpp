#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::tokenizer
{

// Tokenizer 错误码
enum class TokenizerErrorCode : std::uint8_t
{
    InvalidConfiguration = 0,
    InvalidInput = 1,
    UnsupportedFeature = 2,
};

// Tokenizer 错误类型
using TokenizerError = common::Error;

} // namespace liteinfer::core::tokenizer

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<tokenizer::TokenizerErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Tokenizer;
};

} // namespace liteinfer::core::common
