#ifndef _WIN32

#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <expected>
#include <memory>
#include <utility>

#include "core/filesystem/backend/posix/posix_error.hpp"
#include "core/filesystem/backend/posix/posix_file_handle_backend.hpp"

namespace liteinfer::core::filesystem::backend::posix
{

namespace
{

std::expected<void, FilesystemError>
validate_open_request(const std::filesystem::path & path, const FileOpenOptions & options)
{
    const auto & native_path = path.native();
    if (native_path.empty() || native_path.find('\0') != std::filesystem::path::string_type::npos) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::InvalidPath,
                "Open file failed because path is invalid"
            }
        );
    }
    if (options.access != FileAccess::ReadOnly ||
        options.create_mode != FileCreateMode::OpenExisting) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::InvalidArgument,
                "Open file failed because options are invalid"
            }
        );
    }

    return {};
}

} // namespace

std::expected<FileHandle, FilesystemError>
PosixFilesystemBackend::open(const std::filesystem::path & path, const FileOpenOptions & options)
{
    auto validation = validate_open_request(path, options);
    if (!validation.has_value()) {
        return std::unexpected(std::move(validation.error()));
    }

    const int flags = O_RDONLY | O_CLOEXEC;
    int fd = -1;

    do {
        fd = ::open(path.c_str(), flags);
    } while (fd < 0 && errno == EINTR);

    if (fd < 0) {
        return std::unexpected(FilesystemError {detail::map_errno(errno), "Open file failed"});
    }

    PosixFileHandleBackend native_handle(fd);
    struct stat stat_buffer {};
    if (::fstat(fd, &stat_buffer) != 0) {
        return std::unexpected(
            FilesystemError {
                detail::map_errno(errno),
                "Open file failed while inspecting file type"
            }
        );
    }
    if (S_ISDIR(stat_buffer.st_mode)) {
        return std::unexpected(
            FilesystemError {
                FilesystemErrorCode::NotAFile,
                "Open file failed because path refers to a directory"
            }
        );
    }

    auto file_backend = std::make_unique<PosixFileHandleBackend>(std::move(native_handle));
    return FileHandle::create(std::move(file_backend));
}

std::expected<bool, FilesystemError> PosixFilesystemBackend::exists(
    const std::filesystem::path & path
)
{
    std::error_code error;
    const bool result = std::filesystem::exists(path, error);
    if (error) {
        return std::unexpected(
            FilesystemError {detail::map_errno(error.value()), "Check file existence failed"}
        );
    }
    return result;
}

} // namespace liteinfer::core::filesystem::backend::posix

#endif // !_WIN32
