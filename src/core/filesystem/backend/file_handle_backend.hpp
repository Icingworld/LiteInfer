#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem::backend
{

// 文件句柄后端接口基类
class FileHandleBackend
{
public:
    virtual ~FileHandleBackend() = default;

public:
    // 关闭文件，如果关闭失败则返回错误
    // 该操作是幂等的，不会重复关闭文件
    [[nodiscard]]
    virtual std::expected<void, FilesystemError> close() = 0;

    // 从指定偏移读取数据，如果读取失败则返回错误
    // 返回实际读取的字节数；到达文件末尾时允许小于缓冲区大小
    [[nodiscard]]
    virtual std::expected<std::size_t, FilesystemError>
    read_at(std::uint64_t offset, std::span<std::byte> buffer) = 0;

    // 获取文件大小，如果获取失败则返回错误
    [[nodiscard]]
    virtual std::expected<std::uint64_t, FilesystemError> size() = 0;
};

} // namespace liteinfer::core::filesystem::backend
