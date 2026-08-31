#pragma once

#ifndef _WIN32

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend::posix
{

// POSIX 文件句柄后端实现
class PosixFileHandleBackend final : public FileHandleBackend
{
public:
    explicit PosixFileHandleBackend(int fd) noexcept;

    PosixFileHandleBackend(const PosixFileHandleBackend &) = delete;

    PosixFileHandleBackend &operator=(const PosixFileHandleBackend &) = delete;

    PosixFileHandleBackend(PosixFileHandleBackend && other) noexcept;

    PosixFileHandleBackend &operator=(PosixFileHandleBackend && other) noexcept;

    // 析构时会关闭文件，可能会失败
    ~PosixFileHandleBackend() override;

public:
    // 关闭文件，如果关闭失败则返回错误
    // 该操作是幂等的，不会重复关闭文件
    [[nodiscard]]
    std::expected<void, FilesystemError> close() override;

    // 从指定偏移读取数据，如果读取失败则返回错误
    // 返回实际读取的字节数；到达文件末尾时允许小于缓冲区大小
    [[nodiscard]]
    std::expected<std::size_t, FilesystemError> read_at(std::uint64_t offset, std::span<std::byte> buffer) override;

    // 获取文件大小，如果获取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint64_t, FilesystemError> size() override;

private:
    int fd_;
};

} // namespace liteinfer::core::filesystem::backend::posix

#endif // !_WIN32
