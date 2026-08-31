#pragma once

#include <expected>
#include <filesystem>

#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/file_options.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend
{

// 文件系统后端接口基类
class FilesystemBackend
{
public:
    virtual ~FilesystemBackend() = default;

public:
    // 打开文件，如果打开失败则返回错误
    [[nodiscard]]
    virtual std::expected<FileHandle, FilesystemError> open(const std::filesystem::path & path, const FileOpenOptions & options) = 0;

    // 检查文件是否存在，如果检查失败则返回错误
    [[nodiscard]]
    virtual std::expected<bool, FilesystemError> exists(const std::filesystem::path & path) = 0;
};

} // namespace liteinfer::core::filesystem::backend
