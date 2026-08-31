#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem
{

// 文件句柄
// 如果文件句柄持有的后端已经被释放，则所有操作将会返回 ClosedHandle 错误
// 如果文件句柄持有的后端已经被移动，则所有操作将会返回 InvalidState 错误
class FileHandle
{
public:
    FileHandle(const FileHandle &) = delete;

    FileHandle & operator=(const FileHandle &) = delete;

    FileHandle(FileHandle && other) noexcept;

    FileHandle & operator=(FileHandle && other) noexcept;

    ~FileHandle();

private:
    explicit FileHandle(std::unique_ptr<backend::FileHandleBackend> backend);

public:
    // 创建文件句柄，如果创建失败则返回错误
    [[nodiscard]]
    static std::expected<FileHandle, FilesystemError> create(
        std::unique_ptr<backend::FileHandleBackend> backend
    );

    // 关闭文件，如果关闭失败则返回错误
    // 该操作是幂等的，不会重复关闭文件
    [[nodiscard]]
    std::expected<void, FilesystemError> close();

    // 从指定偏移读取数据，如果读取失败则返回错误
    // 返回实际读取的字节数；到达文件末尾时允许小于缓冲区大小
    [[nodiscard]]
    std::expected<std::size_t, FilesystemError>
    read_at(std::uint64_t offset, std::span<std::byte> buffer);

    // 获取文件大小，如果获取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint64_t, FilesystemError> size();

private:
    std::unique_ptr<backend::FileHandleBackend> backend_;
};

} // namespace liteinfer::core::filesystem
