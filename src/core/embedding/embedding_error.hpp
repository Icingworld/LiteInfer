#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::embedding
{

// 嵌入层错误码
enum class EmbeddingErrorCode : std::uint8_t
{
    InvalidWeightMatrix = 0,
    InvalidInputRank = 1,
    InvalidInput = 2,
    InvalidTokenId = 3,
    UnsupportedInputDataType = 4,
};

// 嵌入层错误类型
using EmbeddingError = common::Error;

} // namespace liteinfer::core::embedding

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<embedding::EmbeddingErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Embedding;
};

} // namespace liteinfer::core::common
