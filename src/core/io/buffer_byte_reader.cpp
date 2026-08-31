#include "core/io/buffer_byte_reader.hpp"

#include <algorithm>
#include <cstring>

namespace liteinfer::core::io
{

BufferByteReader::BufferByteReader(std::span<const std::byte> data) noexcept
    : data_(data)
    , offset_(0)
{}

std::expected<std::size_t, IoError> BufferByteReader::read_some(std::span<std::byte> data)
{
    if (data.empty()) {
        return 0;
    }

    const auto remaining = data_.size() - offset_;
    const auto count = std::min(remaining, data.size());
    if (count != 0) {
        std::memcpy(data.data(), data_.data() + offset_, count);
    }
    offset_ += count;
    return count;
}

} // namespace liteinfer::core::io
