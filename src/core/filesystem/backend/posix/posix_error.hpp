#pragma once

#ifndef _WIN32

#include <cerrno>

#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend::posix::detail
{

// 将 POSIX errno 映射为稳定的跨后端错误码。
inline FilesystemErrorCode map_errno(int error) noexcept
{
    switch (error) {
    case ENOENT:
        return FilesystemErrorCode::NotFound;
    case EEXIST:
        return FilesystemErrorCode::AlreadyExists;
    case EACCES:
    case EPERM:
        return FilesystemErrorCode::PermissionDenied;
    case EBADF:
        return FilesystemErrorCode::ClosedHandle;
    case EFAULT:
    case EINVAL:
        return FilesystemErrorCode::InvalidArgument;
    case ELOOP:
    case ENAMETOOLONG:
        return FilesystemErrorCode::InvalidPath;
    case ENOTDIR:
        return FilesystemErrorCode::NotADirectory;
    case EISDIR:
        return FilesystemErrorCode::NotAFile;
    case ENOTEMPTY:
        return FilesystemErrorCode::DirectoryNotEmpty;
    case EROFS:
        return FilesystemErrorCode::ReadOnly;
#if defined(EDQUOT)
    case EDQUOT:
#endif
    case ENOSPC:
        return FilesystemErrorCode::NoSpace;
    case EBUSY:
    case EMFILE:
    case ENFILE:
        return FilesystemErrorCode::ResourceBusy;
    case ENOSYS:
#if defined(ENOTSUP)
    case ENOTSUP:
#endif
#if defined(EOPNOTSUPP) && (!defined(ENOTSUP) || EOPNOTSUPP != ENOTSUP)
    case EOPNOTSUPP:
#endif
        return FilesystemErrorCode::Unsupported;
    default:
        return FilesystemErrorCode::IoError;
    }
}

} // namespace liteinfer::core::filesystem::backend::posix::detail

#endif // !_WIN32
