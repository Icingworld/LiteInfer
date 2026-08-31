#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "core/filesystem/file_handle.hpp"
#include "core/io/byte_reader.hpp"
#include "core/io/io_error.hpp"

namespace liteinfer::core::io
{

// 基于文件偏移的字节读取器
// 借用传入的 FileHandle，读取器不得比文件句柄存活更久
// 逻辑偏移仅支持单消费者访问，不保证多线程共享安全
class FileByteReader final : public ByteReader
{
public:
    explicit FileByteReader(filesystem::FileHandle & file) noexcept;

public:
    // 读取字节数据
    [[nodiscard]]
    std::expected<std::size_t, IoError> read_some(std::span<std::byte> data) override;

    // 获取文件偏移
    [[nodiscard]]
    std::uint64_t offset() const noexcept;

private:
    filesystem::FileHandle * file_;
    std::uint64_t offset_;
};

} // namespace liteinfer::core::io
