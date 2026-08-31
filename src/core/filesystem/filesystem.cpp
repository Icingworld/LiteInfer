#include "core/filesystem/filesystem.hpp"

#include <utility>

#include "core/filesystem/backend/filesystem_backend.hpp"

namespace liteinfer::core::filesystem
{

Filesystem::Filesystem(std::unique_ptr<backend::FilesystemBackend> backend) noexcept
    : backend_(std::move(backend))
{}

Filesystem::Filesystem(Filesystem && other) noexcept
    : backend_(std::move(other.backend_))
{}

Filesystem &Filesystem::operator=(Filesystem && other) noexcept
{
    if (this != &other) {
        backend_ = std::move(other.backend_);
    }
    return *this;
}

Filesystem::~Filesystem() = default;

std::expected<Filesystem, FilesystemError> Filesystem::create(std::unique_ptr<backend::FilesystemBackend> backend)
{
    if (backend == nullptr) [[unlikely]] {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Create filesystem failed because backend is invalid"
        });
    }
    return Filesystem(std::move(backend));
}

std::expected<FileHandle, FilesystemError> Filesystem::open(const std::filesystem::path & path, const FileOpenOptions & options)
{
    if (backend_ == nullptr) [[unlikely]] {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Open file failed because filesystem is invalid"
        });
    }
    return backend_->open(path, options);
}

std::expected<bool, FilesystemError> Filesystem::exists(const std::filesystem::path & path)
{
    if (backend_ == nullptr) [[unlikely]] {
        return std::unexpected(FilesystemError {
            FilesystemErrorCode::InvalidState,
            "Check file existence failed because filesystem is invalid"
        });
    }
    return backend_->exists(path);
}

} // namespace liteinfer::core::filesystem
