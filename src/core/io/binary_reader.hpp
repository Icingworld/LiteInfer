#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "core/io/byte_reader.hpp"
#include "core/io/io_error.hpp"

namespace liteinfer::core::io
{

// 二进制解码资源限制
struct BinaryDecodeLimits
{
    std::uint64_t max_total_bytes; // 最多可读取的总字节数
    std::uint32_t max_string_bytes; // 单个字符串的最大字节数
};

// 有界二进制读取器
// 通过模板传入字节序，支持 std::endian::little 和 std::endian::big
template <std::endian E>
class BasicBinaryReader
{
public:
    BasicBinaryReader(ByteReader & reader, BinaryDecodeLimits limits) noexcept;

public:
    // 读取一个 8 位无符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint8_t, IoError> read_u8();

    // 读取一个 16 位无符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint16_t, IoError> read_u16();

    // 读取一个 32 位无符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint32_t, IoError> read_u32();

    // 读取一个 64 位无符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::uint64_t, IoError> read_u64();

    // 读取一个 32 位有符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::int32_t, IoError> read_i32();

    // 读取一个 64 位有符号整数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::int64_t, IoError> read_i64();

    // 读取一个 32 位浮点数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<float, IoError> read_f32();

    // 读取一个 64 位浮点数，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<double, IoError> read_f64();

    // 读取一个字符串，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<std::string, IoError> read_string();

    // 获取剩余读取预算，如果读取失败则返回错误
    [[nodiscard]]
    std::uint64_t remaining_bytes() const noexcept;

private:
    // 读取精确的字节数据，如果读取失败则返回错误
    [[nodiscard]]
    std::expected<void, IoError> read_exact_bytes(void * data, std::size_t size);

private:
    ByteReader & reader_;
    BinaryDecodeLimits limits_;
    std::uint64_t remaining_bytes_;
};

// 小端二进制读取器
using LittleEndianBinaryReader = BasicBinaryReader<std::endian::little>;

// 大端二进制读取器
using BigEndianBinaryReader = BasicBinaryReader<std::endian::big>;

extern template class BasicBinaryReader<std::endian::little>;
extern template class BasicBinaryReader<std::endian::big>;

} // namespace liteinfer::core::io
