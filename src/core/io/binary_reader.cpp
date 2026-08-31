#include "core/io/binary_reader.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

namespace liteinfer::core::io
{

namespace
{

// 无符号整数类型别名
// 该别名的作用是得到与 T 宽度相同的无符号整数类型
// 例如：std::int32_t -> std::uint32_t
template <std::integral T>
using UnsignedInteger = std::make_unsigned_t<T>;

// 解码字节数组为整数
template <std::endian E, std::integral T>
T decode_integer(const std::array<std::byte, sizeof(T)> & bytes) noexcept
{
    UnsignedInteger<T> value = 0;
    constexpr std::size_t last = sizeof(T) - 1;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const std::size_t position = (E == std::endian::little) ? index : last - index;
        value |= static_cast<UnsignedInteger<T>>(std::to_integer<std::uint8_t>(bytes[position]))
                 << (index * 8U);
    }
    if constexpr (std::is_signed_v<T>) {
        return std::bit_cast<T>(value);
    }
    return value;
}

} // namespace

template <std::endian E>
BasicBinaryReader<E>::BasicBinaryReader(ByteReader & reader, BinaryDecodeLimits limits) noexcept
    : reader_(reader)
    , limits_(limits)
    , remaining_bytes_(limits.max_total_bytes)
{}

template <std::endian E>
std::expected<std::uint8_t, IoError> BasicBinaryReader<E>::read_u8()
{
    std::uint8_t value = 0;
    if (auto read = read_exact_bytes(&value, sizeof(value)); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return value;
}

template <std::endian E>
std::expected<std::uint16_t, IoError> BasicBinaryReader<E>::read_u16()
{
    std::array<std::byte, sizeof(std::uint16_t)> bytes {};
    if (auto read = read_exact_bytes(bytes.data(), bytes.size()); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return decode_integer<E, std::uint16_t>(bytes);
}

template <std::endian E>
std::expected<std::uint32_t, IoError> BasicBinaryReader<E>::read_u32()
{
    std::array<std::byte, sizeof(std::uint32_t)> bytes {};
    if (auto read = read_exact_bytes(bytes.data(), bytes.size()); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return decode_integer<E, std::uint32_t>(bytes);
}

template <std::endian E>
std::expected<std::uint64_t, IoError> BasicBinaryReader<E>::read_u64()
{
    std::array<std::byte, sizeof(std::uint64_t)> bytes {};
    if (auto read = read_exact_bytes(bytes.data(), bytes.size()); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return decode_integer<E, std::uint64_t>(bytes);
}

template <std::endian E>
std::expected<std::int32_t, IoError> BasicBinaryReader<E>::read_i32()
{
    std::array<std::byte, sizeof(std::int32_t)> bytes {};
    if (auto read = read_exact_bytes(bytes.data(), bytes.size()); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return decode_integer<E, std::int32_t>(bytes);
}

template <std::endian E>
std::expected<std::int64_t, IoError> BasicBinaryReader<E>::read_i64()
{
    std::array<std::byte, sizeof(std::int64_t)> bytes {};
    if (auto read = read_exact_bytes(bytes.data(), bytes.size()); !read) [[unlikely]]
    {
        return std::unexpected(std::move(read.error()));
    }
    return decode_integer<E, std::int64_t>(bytes);
}

template <std::endian E>
std::expected<float, IoError> BasicBinaryReader<E>::read_f32()
{
    auto bits = read_u32();
    if (!bits) [[unlikely]] {
        return std::unexpected(std::move(bits.error()));
    }
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    return std::bit_cast<float>(*bits);
}

template <std::endian E>
std::expected<double, IoError> BasicBinaryReader<E>::read_f64()
{
    auto bits = read_u64();
    if (!bits) [[unlikely]] {
        return std::unexpected(std::move(bits.error()));
    }
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    return std::bit_cast<double>(*bits);
}

template <std::endian E>
std::expected<std::string, IoError> BasicBinaryReader<E>::read_string()
{
    auto size = read_u32();
    if (!size) [[unlikely]] {
        return std::unexpected(std::move(size.error()));
    }
    if (*size > limits_.max_string_bytes) [[unlikely]] {
        return std::unexpected(
            IoError {IoErrorCode::ValueTooLarge, "string exceeds the configured decode limit"}
        );
    }
    if (*size > remaining_bytes_) [[unlikely]] {
        return std::unexpected(
            IoError {IoErrorCode::UnexpectedEof, "string length exceeds the remaining binary data"}
        );
    }

    std::string value(*size, '\0');
    if (!value.empty()) {
        if (auto read = read_exact_bytes(value.data(), value.size()); !read) [[unlikely]]
        {
            return std::unexpected(std::move(read.error()));
        }
    }
    return value;
}

template <std::endian E>
std::uint64_t BasicBinaryReader<E>::remaining_bytes() const noexcept
{
    return remaining_bytes_;
}

template <std::endian E>
std::expected<void, IoError> BasicBinaryReader<E>::read_exact_bytes(void * data, std::size_t size)
{
    if (size > remaining_bytes_) [[unlikely]] {
        return std::unexpected(
            IoError {IoErrorCode::UnexpectedEof, "read exceeds the configured binary data budget"}
        );
    }
    auto result = reader_.read_exact(
        std::span {
            static_cast<std::byte *>(data),
            size,
        }
    );
    if (!result) [[unlikely]] {
        return std::unexpected(std::move(result.error()));
    }
    remaining_bytes_ -= size;
    return {};
}

template class BasicBinaryReader<std::endian::little>;
template class BasicBinaryReader<std::endian::big>;

} // namespace liteinfer::core::io
