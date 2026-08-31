#pragma once

#ifndef _WIN32

#error "The Windows filesystem backend is only available on Windows"

#else

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend::windows::detail
{

// 将 Win32 错误码映射为稳定的跨后端错误码
inline FilesystemErrorCode map_error(DWORD error) noexcept
{
    switch (error) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_DRIVE:
    case ERROR_BAD_NETPATH:
    case ERROR_BAD_NET_NAME:
        return FilesystemErrorCode::NotFound;
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS:
        return FilesystemErrorCode::AlreadyExists;
    case ERROR_ACCESS_DENIED:
    case ERROR_NETWORK_ACCESS_DENIED:
    case ERROR_CANNOT_MAKE:
    case ERROR_PRIVILEGE_NOT_HELD:
        return FilesystemErrorCode::PermissionDenied;
    case ERROR_INVALID_HANDLE:
        return FilesystemErrorCode::ClosedHandle;
    case ERROR_INVALID_PARAMETER:
    case ERROR_NEGATIVE_SEEK:
        return FilesystemErrorCode::InvalidArgument;
    case ERROR_INVALID_NAME:
    case ERROR_BAD_PATHNAME:
    case ERROR_FILENAME_EXCED_RANGE:
    case ERROR_BUFFER_OVERFLOW:
        return FilesystemErrorCode::InvalidPath;
    case ERROR_DIRECTORY:
        return FilesystemErrorCode::NotADirectory;
    case ERROR_DIR_NOT_EMPTY:
        return FilesystemErrorCode::DirectoryNotEmpty;
    case ERROR_WRITE_PROTECT:
        return FilesystemErrorCode::ReadOnly;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
        return FilesystemErrorCode::NoSpace;
    case ERROR_BUSY:
    case ERROR_LOCK_VIOLATION:
    case ERROR_SHARING_VIOLATION:
    case ERROR_TOO_MANY_OPEN_FILES:
    case ERROR_NO_SYSTEM_RESOURCES:
        return FilesystemErrorCode::ResourceBusy;
    case ERROR_CALL_NOT_IMPLEMENTED:
    case ERROR_NOT_SUPPORTED:
        return FilesystemErrorCode::Unsupported;
    default:
        return FilesystemErrorCode::IoError;
    }
}

} // namespace liteinfer::core::filesystem::backend::windows::detail

#endif // _WIN32
