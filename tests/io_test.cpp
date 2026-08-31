#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/error.hpp"
#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/filesystem_error.hpp"
#include "core/io/binary_reader.hpp"
#include "core/io/buffer_byte_reader.hpp"
#include "core/io/byte_reader.hpp"
#include "core/io/file_byte_reader.hpp"
#include "core/io/io_error.hpp"

namespace
{

using namespace liteinfer::core;
using filesystem::FileHandle;
using filesystem::FilesystemError;
using filesystem::FilesystemErrorCode;
using io::BigEndianBinaryReader;
using io::BinaryDecodeLimits;
using io::BufferByteReader;
using io::ByteReader;
using io::FileByteReader;
using io::IoError;
using io::IoErrorCode;
using io::LittleEndianBinaryReader;

void assert_io_error(const IoError & error, IoErrorCode expected)
{
    assert(error.category() == common::ErrorCategory::Io);
    assert(error.code() == std::to_underlying(expected));
}

void assert_filesystem_error(const IoError & error, FilesystemErrorCode expected)
{
    assert(error.category() == common::ErrorCategory::Filesystem);
    assert(error.code() == std::to_underlying(expected));
}

template <std::endian E, std::integral T>
void append_integer(std::vector<std::byte> & bytes, T value)
{
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    constexpr std::size_t last = sizeof(T) - 1;
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(T));
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        const auto position = E == std::endian::little ? index : last - index;
        bytes[begin + position] = static_cast<std::byte>((bits >> (index * 8U)) & 0xffU);
    }
}

template <std::endian E>
void append_string(std::vector<std::byte> & bytes, std::string_view value)
{
    append_integer<E>(bytes, static_cast<std::uint32_t>(value.size()));
    for (const char character : value) {
        bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    }
}

class ChunkedByteReader final : public ByteReader
{
public:
    ChunkedByteReader(std::span<const std::byte> data, std::size_t max_chunk) noexcept
        : data_(data)
        , max_chunk_(max_chunk)
    {}

    std::expected<std::size_t, IoError> read_some(std::span<std::byte> data) override
    {
        const auto remaining = data_.size() - offset_;
        const auto count = std::min({remaining, data.size(), max_chunk_});
        if (count != 0) {
            std::memcpy(data.data(), data_.data() + offset_, count);
        }
        offset_ += count;
        return count;
    }

private:
    std::span<const std::byte> data_;
    std::size_t max_chunk_;
    std::size_t offset_ {0};
};

class OverreportingByteReader final : public ByteReader
{
public:
    std::expected<std::size_t, IoError> read_some(std::span<std::byte> data) override
    {
        return data.size() + 1;
    }
};

class FakeFileHandleBackend final : public filesystem::backend::FileHandleBackend
{
public:
    FakeFileHandleBackend(std::span<const std::byte> data, std::size_t max_chunk)
        : data_(data.begin(), data.end())
        , max_chunk_(max_chunk)
    {}

    void fail_reads(FilesystemErrorCode error_code)
    {
        error_code_ = error_code;
    }

    void overreport_reads() noexcept
    {
        overreport_ = true;
    }

    [[nodiscard]]
    std::size_t read_calls() const noexcept
    {
        return read_calls_;
    }

    std::expected<void, FilesystemError> close() override
    {
        closed_ = true;
        return {};
    }

    std::expected<std::size_t, FilesystemError>
    read_at(std::uint64_t offset, std::span<std::byte> buffer) override
    {
        ++read_calls_;
        if (error_code_.has_value()) {
            return std::unexpected(FilesystemError {*error_code_, "fake file handle read failed"});
        }
        if (closed_) {
            return std::unexpected(
                FilesystemError {
                    FilesystemErrorCode::ClosedHandle,
                    "fake file handle is closed",
                }
            );
        }
        if (overreport_) {
            return buffer.size() + 1;
        }
        if (offset >= data_.size()) {
            return 0;
        }

        const auto available = data_.size() - static_cast<std::size_t>(offset);
        const auto count = std::min({available, buffer.size(), max_chunk_});
        std::memcpy(buffer.data(), data_.data() + offset, count);
        return count;
    }

    std::expected<std::uint64_t, FilesystemError> size() override
    {
        return data_.size();
    }

private:
    std::vector<std::byte> data_;
    std::size_t max_chunk_;
    std::size_t read_calls_ {0};
    std::optional<FilesystemErrorCode> error_code_;
    bool overreport_ {false};
    bool closed_ {false};
};

void test_byte_reader_exact()
{
    const std::array source {
        std::byte {'L'},
        std::byte {'i'},
        std::byte {'t'},
        std::byte {'e'},
        std::byte {'r'},
    };
    ChunkedByteReader reader(source, 2);
    std::array<std::byte, 5> destination {};

    assert(reader.read_exact(destination).has_value());
    assert(destination == source);
    assert(reader.read_exact(std::span<std::byte> {}).has_value());

    OverreportingByteReader invalid_reader;
    std::array<std::byte, 2> invalid_destination {};
    auto invalid = invalid_reader.read_exact(invalid_destination);
    assert(!invalid.has_value());
    assert_io_error(invalid.error(), IoErrorCode::InvalidData);

    const std::array short_source {std::byte {'a'}, std::byte {'b'}};
    BufferByteReader short_reader(short_source);
    std::array<std::byte, 3> short_destination {};
    auto eof = short_reader.read_exact(short_destination);
    assert(!eof.has_value());
    assert_io_error(eof.error(), IoErrorCode::UnexpectedEof);
}

void test_binary_reader_endianness_and_values()
{
    std::vector<std::byte> little_data;
    append_integer<std::endian::little>(little_data, std::uint8_t {0x7f});
    append_integer<std::endian::little>(little_data, std::uint16_t {0x1234});
    append_integer<std::endian::little>(little_data, std::uint32_t {0x12345678});
    append_integer<std::endian::little>(little_data, std::uint64_t {0x0123456789abcdef});
    append_integer<std::endian::little>(little_data, std::int32_t {-123456});
    append_integer<std::endian::little>(little_data, std::int64_t {-9876543210LL});
    append_integer<std::endian::little>(little_data, std::bit_cast<std::uint32_t>(1.5F));
    append_integer<std::endian::little>(little_data, std::bit_cast<std::uint64_t>(-2.25));
    append_string<std::endian::little>(little_data, "LiteInfer");
    append_string<std::endian::little>(little_data, "");

    BufferByteReader little_source(little_data);
    LittleEndianBinaryReader little_reader(
        little_source,
        BinaryDecodeLimits {little_data.size(), 32}
    );
    auto u8 = little_reader.read_u8();
    auto u16 = little_reader.read_u16();
    auto u32 = little_reader.read_u32();
    auto u64 = little_reader.read_u64();
    auto i32 = little_reader.read_i32();
    auto i64 = little_reader.read_i64();
    auto f32 = little_reader.read_f32();
    auto f64 = little_reader.read_f64();
    auto string = little_reader.read_string();
    auto empty_string = little_reader.read_string();

    assert(u8.has_value() && *u8 == 0x7f);
    assert(u16.has_value() && *u16 == 0x1234);
    assert(u32.has_value() && *u32 == 0x12345678);
    assert(u64.has_value() && *u64 == 0x0123456789abcdef);
    assert(i32.has_value() && *i32 == -123456);
    assert(i64.has_value() && *i64 == -9876543210LL);
    assert(f32.has_value() && *f32 == 1.5F);
    assert(f64.has_value() && *f64 == -2.25);
    assert(string.has_value() && *string == "LiteInfer");
    assert(empty_string.has_value() && empty_string->empty());
    assert(little_reader.remaining_bytes() == 0);

    std::vector<std::byte> big_data;
    append_integer<std::endian::big>(big_data, std::uint16_t {0x1234});
    append_integer<std::endian::big>(big_data, std::uint32_t {0x12345678});
    append_integer<std::endian::big>(big_data, std::uint64_t {0x0123456789abcdef});
    append_integer<std::endian::big>(big_data, std::int32_t {-123456});
    append_integer<std::endian::big>(big_data, std::bit_cast<std::uint32_t>(1.5F));

    BufferByteReader big_source(big_data);
    BigEndianBinaryReader big_reader(big_source, BinaryDecodeLimits {big_data.size(), 32});
    auto big_u16 = big_reader.read_u16();
    auto big_u32 = big_reader.read_u32();
    auto big_u64 = big_reader.read_u64();
    auto big_i32 = big_reader.read_i32();
    auto big_f32 = big_reader.read_f32();
    assert(big_u16.has_value() && *big_u16 == 0x1234);
    assert(big_u32.has_value() && *big_u32 == 0x12345678);
    assert(big_u64.has_value() && *big_u64 == 0x0123456789abcdef);
    assert(big_i32.has_value() && *big_i32 == -123456);
    assert(big_f32.has_value() && *big_f32 == 1.5F);
    assert(big_reader.remaining_bytes() == 0);
}

void test_binary_reader_limits_and_eof()
{
    std::vector<std::byte> string_data;
    append_string<std::endian::little>(string_data, "abc");

    BufferByteReader too_large_source(string_data);
    LittleEndianBinaryReader too_large_reader(
        too_large_source,
        BinaryDecodeLimits {string_data.size(), 2}
    );
    auto too_large = too_large_reader.read_string();
    assert(!too_large.has_value());
    assert_io_error(too_large.error(), IoErrorCode::ValueTooLarge);
    assert(too_large_reader.remaining_bytes() == string_data.size() - sizeof(std::uint32_t));

    BufferByteReader budget_source(string_data);
    LittleEndianBinaryReader budget_reader(budget_source, BinaryDecodeLimits {3, 32});
    auto budget_error = budget_reader.read_u32();
    assert(!budget_error.has_value());
    assert_io_error(budget_error.error(), IoErrorCode::UnexpectedEof);
    assert(budget_reader.remaining_bytes() == 3);

    BufferByteReader remaining_source(string_data);
    LittleEndianBinaryReader remaining_reader(remaining_source, BinaryDecodeLimits {6, 32});
    auto remaining_error = remaining_reader.read_string();
    assert(!remaining_error.has_value());
    assert_io_error(remaining_error.error(), IoErrorCode::UnexpectedEof);
    assert(remaining_reader.remaining_bytes() == 2);

    std::vector<std::byte> truncated_data;
    append_integer<std::endian::little>(truncated_data, std::uint32_t {4});
    truncated_data.push_back(std::byte {'a'});
    truncated_data.push_back(std::byte {'b'});
    BufferByteReader truncated_source(truncated_data);
    LittleEndianBinaryReader truncated_reader(truncated_source, BinaryDecodeLimits {10, 32});
    auto truncated = truncated_reader.read_string();
    assert(!truncated.has_value());
    assert_io_error(truncated.error(), IoErrorCode::UnexpectedEof);
    assert(truncated_reader.remaining_bytes() == 6);
}

void test_file_byte_reader()
{
    const std::array source {
        std::byte {'L'},
        std::byte {'i'},
        std::byte {'t'},
        std::byte {'e'},
        std::byte {'I'},
        std::byte {'n'},
        std::byte {'f'},
        std::byte {'e'},
    };
    auto backend = std::make_unique<FakeFileHandleBackend>(source, 2);
    auto * backend_pointer = backend.get();
    auto handle_result = FileHandle::create(std::move(backend));
    assert(handle_result.has_value());
    auto handle = std::move(*handle_result);

    FileByteReader reader(handle);
    std::array<std::byte, 8> destination {};
    assert(reader.read_exact(destination).has_value());
    assert(destination == source);
    assert(reader.offset() == source.size());

    std::array<std::byte, 1> eof_buffer {};
    auto eof = reader.read_some(eof_buffer);
    assert(eof.has_value() && *eof == 0);

    const auto calls_before_empty_read = backend_pointer->read_calls();
    assert(reader.read_some(std::span<std::byte> {}).has_value());
    assert(backend_pointer->read_calls() == calls_before_empty_read);

    auto failing_backend = std::make_unique<FakeFileHandleBackend>(source, source.size());
    failing_backend->fail_reads(FilesystemErrorCode::PermissionDenied);
    auto failing_handle_result = FileHandle::create(std::move(failing_backend));
    assert(failing_handle_result.has_value());
    auto failing_handle = std::move(*failing_handle_result);
    FileByteReader failing_reader(failing_handle);
    auto failure = failing_reader.read_some(destination);
    assert(!failure.has_value());
    assert_filesystem_error(failure.error(), FilesystemErrorCode::PermissionDenied);
    assert(failing_reader.offset() == 0);

    auto overreporting_backend = std::make_unique<FakeFileHandleBackend>(source, source.size());
    overreporting_backend->overreport_reads();
    auto overreporting_handle_result = FileHandle::create(std::move(overreporting_backend));
    assert(overreporting_handle_result.has_value());
    auto overreporting_handle = std::move(*overreporting_handle_result);
    FileByteReader overreporting_reader(overreporting_handle);
    auto overreport = overreporting_reader.read_some(destination);
    assert(!overreport.has_value());
    assert_io_error(overreport.error(), IoErrorCode::InvalidData);
    assert(overreporting_reader.offset() == 0);
}

} // namespace

int main()
{
    test_byte_reader_exact();
    test_binary_reader_endianness_and_values();
    test_binary_reader_limits_and_eof();
    test_file_byte_reader();
}
