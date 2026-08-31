#ifdef _WIN32

#include "core/filesystem/backend/windows/windows_file_handle_backend.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include "core/filesystem/backend/windows/windows_error.hpp"

namespace liteinfer::core::filesystem::backend::windows
{

namespace
{

[[nodiscard]]
HANDLE as_handle(void * native_handle) noexcept
{
    return static_cast<HANDLE>(native_handle);
}

[[nodiscard]]
bool is_closed(void * native_handle) noexcept
{
    return native_handle == nullptr || as_handle(native_handle) == INVALID_HANDLE_VALUE;
}

[[nodiscard]]
bool valid_read_range(std::uint64_t offset, std::size_t size) noexcept
{
    constexpr auto max_offset =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    return offset <= max_offset && (size == 0 || size - 1 <= max_offset - offset);
}

class EventGuard final
{
public:
    explicit EventGuard(HANDLE handle) noexcept
        : handle_(handle)
    {}

    EventGuard(const EventGuard &) = delete;
    EventGuard & operator=(const EventGuard &) = delete;

    ~EventGuard()
    {
        if (handle_ != nullptr) {
            static_cast<void>(::CloseHandle(handle_));
        }
    }

    [[nodiscard]]
    HANDLE handle() const noexcept
    {
        return handle_;
    }

private:
    HANDLE handle_;
};

} // namespace

WindowsFileHandleBackend::WindowsFileHandleBackend(void * native_handle) noexcept
    : native_handle_(native_handle)
{}

WindowsFileHandleBackend::WindowsFileHandleBackend(WindowsFileHandleBackend && other) noexcept
    : native_handle_(std::exchange(other.native_handle_, nullptr))
{}

WindowsFileHandleBackend & WindowsFileHandleBackend::operator=(
    WindowsFileHandleBackend && other
) noexcept
{
    if (this != &other) {
        static_cast<void>(close());
        native_handle_ = std::exchange(other.native_handle_, nullptr);
    }
    return *this;
}

WindowsFileHandleBackend::~WindowsFileHandleBackend()
{
    static_cast<void>(close());
}

std::expected<void, FilesystemError> WindowsFileHandleBackend::close()
{
    if (is_closed(native_handle_)) {
        native_handle_ = nullptr;
        return {};
    }

    // CloseHandle 失败后句柄状态不应再被本对象使用，因此先清空所有权。
    void * native_handle = std::exchange(native_handle_, nullptr);
    if (!::CloseHandle(as_handle(native_handle))) {
        return std::unexpected(
            FilesystemError {
                detail::map_error(::GetLastError()),
                "Failed to close file handle",
            }
        );
    }
    return {};
}

std::expected<std::size_t, FilesystemError>
WindowsFileHandleBackend::read_at(std::uint64_t offset, std::span<std::byte> buffer)
{
    if (is_closed(native_handle_)) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::ClosedHandle,
                "Failed to read_at because file handle is closed",
            }
        );
    }
    if (!valid_read_range(offset, buffer.size())) {
        return std::unexpected(
            FilesystemError {FilesystemErrorCode::InvalidArgument, "Invalid offset or buffer size"}
        );
    }

    constexpr auto max_chunk_size = std::numeric_limits<DWORD>::max();
    std::size_t read_total = 0;
    while (read_total < buffer.size()) {
        const auto chunk_size = std::min<std::size_t>(
            buffer.size() - read_total,
            static_cast<std::size_t>(max_chunk_size)
        );
        const auto read_offset = offset + read_total;

        OVERLAPPED overlapped {};
        EventGuard event_guard(::CreateEventW(nullptr, TRUE, FALSE, nullptr));
        if (event_guard.handle() == nullptr) {
            return std::unexpected(
                FilesystemError {
                    detail::map_error(::GetLastError()),
                    "Failed to create read event",
                }
            );
        }
        overlapped.hEvent = event_guard.handle();
        overlapped.Offset = static_cast<DWORD>(read_offset & 0xffffffffULL);
        overlapped.OffsetHigh = static_cast<DWORD>(read_offset >> 32U);

        DWORD bytes_read = 0;
        const BOOL read_succeeded = ::ReadFile(
            as_handle(native_handle_),
            buffer.data() + read_total,
            static_cast<DWORD>(chunk_size),
            &bytes_read,
            &overlapped
        );
        if (!read_succeeded) {
            DWORD error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) {
                break;
            }
            if (error == ERROR_IO_PENDING) {
                if (!::GetOverlappedResult(
                        as_handle(native_handle_),
                        &overlapped,
                        &bytes_read,
                        TRUE
                    )) {
                    error = ::GetLastError();
                    if (error == ERROR_HANDLE_EOF) {
                        break;
                    }
                    return std::unexpected(
                        FilesystemError {detail::map_error(error), "Failed to read from file"}
                    );
                }
            } else {
                return std::unexpected(
                    FilesystemError {detail::map_error(error), "Failed to read from file"}
                );
            }
        }

        if (bytes_read > chunk_size) {
            return std::unexpected(
                FilesystemError {
                    FilesystemErrorCode::IoError,
                    "Failed to read from file because the byte count is invalid",
                }
            );
        }
        read_total += static_cast<std::size_t>(bytes_read);
        if (bytes_read < chunk_size) {
            break;
        }
    }

    return read_total;
}

std::expected<std::uint64_t, FilesystemError> WindowsFileHandleBackend::size()
{
    if (is_closed(native_handle_)) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::ClosedHandle,
                "Failed to get file size because file handle is closed",
            }
        );
    }

    LARGE_INTEGER file_size {};
    if (!::GetFileSizeEx(as_handle(native_handle_), &file_size)) {
        return std::unexpected(
            FilesystemError {detail::map_error(::GetLastError()), "Failed to get file size"}
        );
    }
    if (file_size.QuadPart < 0) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::IoError,
                "Failed to get file size because the reported size is negative",
            }
        );
    }

    return static_cast<std::uint64_t>(file_size.QuadPart);
}

} // namespace liteinfer::core::filesystem::backend::windows

#endif // _WIN32
