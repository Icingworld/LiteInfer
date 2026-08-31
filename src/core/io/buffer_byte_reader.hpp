#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/io/byte_reader.hpp"
#include "core/io/io_error.hpp"

namespace liteinfer::core::io
{

// 内存缓冲区字节读取器
class BufferByteReader final : public ByteReader
{
public:
    explicit BufferByteReader(std::span<const std::byte> data) noexcept;

public:
    // 尽力读取字节数据
    // 允许短读，返回的字节数可能小于请求的字节数
    [[nodiscard]]
    std::expected<std::size_t, IoError> read_some(std::span<std::byte> data) override;

private:
    std::span<const std::byte> data_;
    std::size_t offset_;
};

} // namespace liteinfer::core::io
