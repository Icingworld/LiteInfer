#pragma once

#ifdef _WIN32

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend::windows
{

// Windows 文件句柄后端实现
// native_handle 使用 void * 保存 Win32 HANDLE，避免将 windows.h 暴露给公共接口
class WindowsFileHandleBackend final : public FileHandleBackend
{
public:
    explicit WindowsFileHandleBackend(void * native_handle) noexcept;

    WindowsFileHandleBackend(const WindowsFileHandleBackend &) = delete;

    WindowsFileHandleBackend & operator=(const WindowsFileHandleBackend &) = delete;

    WindowsFileHandleBackend(WindowsFileHandleBackend && other) noexcept;

    WindowsFileHandleBackend & operator=(WindowsFileHandleBackend && other) noexcept;

    // 析构时会关闭文件，可能会失败
    ~WindowsFileHandleBackend() override;

public:
    // 关闭文件，如果关闭失败则返回错误
    // 该操作是幂等的，不会重复关闭文件
    [[nodiscard]]
    std::expected<void, FilesystemError> close() override;

    // 从指定偏移读取数据，如果读取失败则返回错误
    // 返回实际读取的字节数；到达文件末尾时允许小于缓冲区大小
    [[nodiscard]]
    std::expected<std::size_t, FilesystemError>
    read_at(std::uint64_t offset, std::span<std::byte> buffer) override;

    // 获取文件大小，如果获取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint64_t, FilesystemError> size() override;

private:
    void * native_handle_;
};

} // namespace liteinfer::core::filesystem::backend::windows

#endif // _WIN32
