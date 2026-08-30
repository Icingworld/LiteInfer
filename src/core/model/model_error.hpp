#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::model
{

// 模型错误码
enum class ModelErrorCode : std::uint8_t
{
    InvalidConfiguration = 0,
};

// 模型错误类型
using ModelError = common::Error;

} // namespace liteinfer::core::model

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<model::ModelErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Model;
};

} // namespace liteinfer::core::common
