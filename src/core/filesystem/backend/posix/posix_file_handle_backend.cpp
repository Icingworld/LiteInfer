#ifndef _WIN32

#include "core/filesystem/backend/posix/posix_file_handle_backend.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <utility>

#include "core/filesystem/backend/posix/posix_error.hpp"

namespace liteinfer::core::filesystem::backend::posix
{

PosixFileHandleBackend::PosixFileHandleBackend(int fd) noexcept
    : fd_(fd)
{}

PosixFileHandleBackend::PosixFileHandleBackend(PosixFileHandleBackend && other) noexcept
    : fd_(std::exchange(other.fd_, -1))
{}

PosixFileHandleBackend & PosixFileHandleBackend::operator=(PosixFileHandleBackend && other) noexcept
{
    if (this != &other) {
        static_cast<void>(close());
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

PosixFileHandleBackend::~PosixFileHandleBackend()
{
    static_cast<void>(close());
}

std::expected<void, FilesystemError> PosixFileHandleBackend::close()
{
    if (fd_ < 0) {
        return {};
    }

    // 不能保证在 close 失败后，fd 仍然有效，因此先置为 -1
    const int fd = fd_;
    fd_ = -1;

    if (::close(fd) != 0) {
        return std::unexpected(
            FilesystemError {
                detail::map_errno(errno),
                "Failed to close file handle",
            }
        );
    }
    return {};
}

std::expected<std::size_t, FilesystemError>
PosixFileHandleBackend::read_at(std::uint64_t offset, std::span<std::byte> buffer)
{
    if (fd_ < 0) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::ClosedHandle,
                "Failed to read_at because file handle is closed"
            }
        );
    }

    // offset 是 uint64_t，而 off_t 是有符号的，所以需要验证范围
    constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (offset > max_offset || (!buffer.empty() && buffer.size() - 1 > max_offset - offset)) {
        return std::unexpected(
            FilesystemError {FilesystemErrorCode::InvalidArgument, "Invalid offset or buffer size"}
        );
    }

    auto native_offset = static_cast<off_t>(offset);

    std::size_t read_total = 0;
    while (read_total < buffer.size()) {
        const auto chunk_size = std::min<std::size_t>(
            buffer.size() - read_total,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())
        );
        const auto read_offset = native_offset + static_cast<off_t>(read_total);
        const auto read_size = ::pread(fd_, buffer.data() + read_total, chunk_size, read_offset);
        if (read_size < 0) {
            if (errno == EINTR) {
                continue;
            }
            return std::unexpected(
                FilesystemError {detail::map_errno(errno), "Failed to read from file"}
            );
        }
        if (read_size == 0) {
            break;
        }
        read_total += static_cast<std::size_t>(read_size);
    }

    return read_total;
}

std::expected<std::uint64_t, FilesystemError> PosixFileHandleBackend::size()
{
    if (fd_ < 0) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::ClosedHandle,
                "Failed to get file size because file handle is closed"
            }
        );
    }

    struct stat stat_buffer {};
    if (::fstat(fd_, &stat_buffer) != 0) {
        return std::unexpected(
            FilesystemError {detail::map_errno(errno), "Failed to get file size"}
        );
    }

    if (S_ISDIR(stat_buffer.st_mode)) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::NotAFile,
                "Failed to get file size because handle refers to a directory"
            }
        );
    }
    if (stat_buffer.st_size < 0) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::IoError,
                "Failed to get file size because the reported size is negative"
            }
        );
    }

    return static_cast<std::uint64_t>(stat_buffer.st_size);
}

} // namespace liteinfer::core::filesystem::backend::posix

#endif // !_WIN32
