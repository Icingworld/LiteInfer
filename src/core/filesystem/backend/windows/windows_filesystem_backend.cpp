#ifdef _WIN32

#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <expected>
#include <memory>
#include <utility>

#include "core/filesystem/backend/windows/windows_error.hpp"
#include "core/filesystem/backend/windows/windows_file_handle_backend.hpp"

namespace liteinfer::core::filesystem::backend::windows
{

namespace
{

std::expected<void, FilesystemError>
validate_path_and_options(const std::filesystem::path & path, const FileOpenOptions & options)
{
    const auto & native_path = path.native();
    if (native_path.empty() || native_path.find(std::filesystem::path::value_type {}) !=
                                   std::filesystem::path::string_type::npos) [[unlikely]] {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::InvalidPath,
                "Open file failed because path is invalid",
            }
        );
    }
    if (options.access != FileAccess::ReadOnly ||
        options.create_mode != FileCreateMode::OpenExisting) [[unlikely]] {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::InvalidArgument,
                "Open file failed because options are invalid",
            }
        );
    }

    return {};
}

std::expected<DWORD, FilesystemError> get_attributes(const std::filesystem::path & path)
{
    const DWORD attributes = ::GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES) [[likely]] {
        return attributes;
    }
    return std::unexpected(
        FilesystemError {detail::map_error(::GetLastError()), "Get file attributes failed"}
    );
}

} // namespace

std::expected<FileHandle, FilesystemError>
WindowsFilesystemBackend::open(const std::filesystem::path & path, const FileOpenOptions & options)
{
    auto validation = validate_path_and_options(path, options);
    if (!validation.has_value()) [[unlikely]] {
        return std::unexpected(std::move(validation.error()));
    }

    auto attributes = get_attributes(path);
    if (!attributes.has_value()) [[unlikely]] {
        return std::unexpected(std::move(attributes.error()));
    }
    if ((*attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) [[unlikely]] {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::NotAFile,
                "Open file failed because path refers to a directory",
            }
        );
    }

    constexpr DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED;
    HANDLE native_handle = ::CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_EXISTING,
        flags,
        nullptr
    );
    if (native_handle == INVALID_HANDLE_VALUE) [[unlikely]] {
        return std::unexpected(
            FilesystemError {detail::map_error(::GetLastError()), "Open file failed"}
        );
    }

    auto file_backend = std::make_unique<WindowsFileHandleBackend>(native_handle);
    return FileHandle::create(std::move(file_backend));
}

std::expected<bool, FilesystemError> WindowsFilesystemBackend::exists(
    const std::filesystem::path & path
)
{
    const DWORD attributes = ::GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES) [[likely]] {
        return true;
    }

    const DWORD error = ::GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ||
        error == ERROR_INVALID_DRIVE || error == ERROR_BAD_NETPATH || error == ERROR_BAD_NET_NAME) {
        return false;
    }
    return std::unexpected(
        FilesystemError {detail::map_error(error), "Check file existence failed"}
    );
}

} // namespace liteinfer::core::filesystem::backend::windows

#endif // _WIN32
