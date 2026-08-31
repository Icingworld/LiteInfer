#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "core/filesystem/backend/filesystem_backend.hpp"
#include "core/filesystem/filesystem_error.hpp"
#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/file_options.hpp"

namespace liteinfer::core::filesystem
{

// 文件系统
// 如果文件系统后端为空，则所有操作将会返回 InvalidState 错误
class Filesystem
{
public:
    Filesystem(const Filesystem &) = delete;

    Filesystem &operator=(const Filesystem &) = delete;

    Filesystem(Filesystem && other) noexcept;

    Filesystem &operator=(Filesystem && other) noexcept;

    ~Filesystem();

private:
    explicit Filesystem(std::unique_ptr<backend::FilesystemBackend> backend) noexcept;

public:
    // 创建文件系统，如果创建失败则返回错误
    [[nodiscard]]
    static std::expected<Filesystem, FilesystemError> create(std::unique_ptr<backend::FilesystemBackend> backend);

    // 打开文件
    [[nodiscard]]
    std::expected<FileHandle, FilesystemError> open(const std::filesystem::path & path, const FileOpenOptions & options = {});

    // 检查文件是否存在
    [[nodiscard]]
    std::expected<bool, FilesystemError> exists(const std::filesystem::path & path);

private:
    std::unique_ptr<backend::FilesystemBackend> backend_;
};

} // namespace liteinfer::core::filesystem
