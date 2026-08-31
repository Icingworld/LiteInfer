#include "core/filesystem/file_handle.hpp"

#include <utility>

#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace liteinfer::core::filesystem
{

FileHandle::FileHandle(std::unique_ptr<backend::FileHandleBackend> backend)
    : backend_(std::move(backend))
{
}

FileHandle::FileHandle(FileHandle && other) noexcept = default;

FileHandle &FileHandle::operator=(FileHandle && other) noexcept
{
    if (this != &other) {
        backend_ = std::move(other.backend_);
    }
    return *this;
}

FileHandle::~FileHandle() = default;

std::expected<FileHandle, FilesystemError> FileHandle::create(std::unique_ptr<backend::FileHandleBackend> backend)
{
    if (backend == nullptr) {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Create file handle failed because backend is invalid"
        });
    }
    return FileHandle(std::move(backend));
}

std::expected<void, FilesystemError> FileHandle::close()
{
    if (backend_ == nullptr) {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Close file failed because backend is invalid"
        });
    }
    return backend_->close();
}

std::expected<std::size_t, FilesystemError> FileHandle::read_at(std::uint64_t offset, std::span<std::byte> buffer)
{
    if (backend_ == nullptr) {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Read file failed because backend is invalid"
        });
    }
    return backend_->read_at(offset, buffer);
}

std::expected<std::uint64_t, FilesystemError> FileHandle::size()
{
    if (backend_ == nullptr) {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Get size failed because backend is invalid"
        });
    }
    return backend_->size();
}

} // namespace liteinfer::core::filesystem
