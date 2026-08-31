#pragma once

#ifdef _WIN32

#include <expected>
#include <filesystem>

#include "core/filesystem/backend/filesystem_backend.hpp"
#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/file_options.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend::windows
{

// Windows 文件系统后端实现
class WindowsFilesystemBackend final : public FilesystemBackend
{
public:
    // 打开文件，如果打开失败则返回错误
    [[nodiscard]]
    std::expected<FileHandle, FilesystemError>
    open(const std::filesystem::path & path, const FileOpenOptions & options) override;

    // 检查文件是否存在，如果检查失败则返回错误
    [[nodiscard]]
    std::expected<bool, FilesystemError> exists(const std::filesystem::path & path) override;
};

} // namespace liteinfer::core::filesystem::backend::windows

#endif // _WIN32
