#include "core/io/byte_reader.hpp"

#include <utility>

#include "core/io/io_error.hpp"

namespace liteinfer::core::io
{

std::expected<void, IoError> ByteReader::read_exact(std::span<std::byte> data)
{
    while (!data.empty()) {
        auto read = read_some(data);
        if (!read) [[unlikely]] {
            return std::unexpected(std::move(read.error()));
        }
        if (*read == 0) [[unlikely]] {
            return std::unexpected(
                IoError {IoErrorCode::UnexpectedEof, "unexpected end of binary data"}
            );
        }
        if (*read > data.size()) [[unlikely]] {
            return std::unexpected(
                IoError {IoErrorCode::InvalidData, "byte reader returned more bytes than requested"}
            );
        }
        data = data.subspan(*read);
    }
    return {};
}

} // namespace liteinfer::core::io
