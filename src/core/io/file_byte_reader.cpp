#include "core/io/file_byte_reader.hpp"

#include <limits>
#include <utility>

namespace liteinfer::core::io
{

FileByteReader::FileByteReader(filesystem::FileHandle & file) noexcept
    : file_(&file)
    , offset_(0)
{}

std::expected<std::size_t, IoError> FileByteReader::read_some(std::span<std::byte> data)
{
    if (data.empty()) {
        return 0;
    }

    auto result = file_->read_at(offset_, data);
    if (!result) [[unlikely]] {
        return std::unexpected(std::move(result.error()));
    }
    const auto read = *result;
    if (read > data.size()) [[unlikely]] {
        return std::unexpected(
            IoError {IoErrorCode::InvalidData, "file handle returned more bytes than requested"}
        );
    }
    if (read > std::numeric_limits<std::uint64_t>::max() - offset_) [[unlikely]] {
        return std::unexpected(
            IoError {
                IoErrorCode::InvalidData,
                "file reader returned a byte count that overflows its offset",
            }
        );
    }
    offset_ += read;
    return read;
}

std::uint64_t FileByteReader::offset() const noexcept
{
    return offset_;
}

} // namespace liteinfer::core::io
