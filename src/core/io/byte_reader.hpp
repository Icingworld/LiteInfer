#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/io/io_error.hpp"

namespace liteinfer::core::io
{

// 字节读取器
class ByteReader
{
public:
    virtual ~ByteReader() = default;

public:
    // 尽力读取字节数据
    // 允许短读，返回的字节数可能小于请求的字节数
    [[nodiscard]]
    virtual std::expected<std::size_t, IoError> read_some(std::span<std::byte> data) = 0;

    // 读取精确的字节数据
    // 必须把缓冲区填满，如果读取的字节数不足，则返回错误
    [[nodiscard]]
    std::expected<void, IoError> read_exact(std::span<std::byte> data);
};

} // namespace liteinfer::core::io
