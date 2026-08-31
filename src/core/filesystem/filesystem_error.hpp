#pragma once

#include <cstdint>

#include "core/common/error.hpp"

namespace liteinfer::core::filesystem
{

// 文件系统错误码
enum class FilesystemErrorCode : std::uint8_t
{
    InvalidState = 0,
    ClosedHandle = 1,
    NotFound = 2,
    AlreadyExists = 3,
    PermissionDenied = 4,
    InvalidArgument = 5,
    InvalidPath = 6,
    NotADirectory = 7,
    NotAFile = 8,
    DirectoryNotEmpty = 9,
    ReadOnly = 10,
    NoSpace = 11,
    ResourceBusy = 12,
    IoError = 13,
    Unsupported = 14,
};

// 文件系统错误类型
using FilesystemError = common::Error;

} // namespace liteinfer::core::filesystem

namespace liteinfer::core::common
{

template <>
struct ErrorTraits<filesystem::FilesystemErrorCode>
{
    static constexpr ErrorCategory category = ErrorCategory::Filesystem;
};

} // namespace liteinfer::core::common
