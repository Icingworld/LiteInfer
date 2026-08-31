#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "core/common/error.hpp"
#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif
#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/file_options.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/filesystem/filesystem_error.hpp"

namespace
{

using namespace liteinfer::core;
using namespace liteinfer::core::filesystem;
#if defined(_WIN32)
using filesystem::backend::windows::WindowsFilesystemBackend;
#else
using filesystem::backend::posix::PosixFilesystemBackend;
#endif

constexpr std::string_view FILE_CONTENT = "LiteInfer";

class TemporaryFixture
{
public:
    TemporaryFixture()
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_filesystem_test_" +
#if defined(_WIN32)
               std::to_string(static_cast<unsigned long long>(::_getpid()))
#else
               std::to_string(static_cast<unsigned long long>(::getpid()))
#endif
              )
          )
        , file_(directory_ / "model.bin")
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        assert(!error);
        assert(std::filesystem::create_directory(directory_));

        std::ofstream stream(file_, std::ios::binary);
        assert(stream.is_open());
        stream.write(FILE_CONTENT.data(), static_cast<std::streamsize>(FILE_CONTENT.size()));
        assert(stream.good());
    }

    TemporaryFixture(const TemporaryFixture &) = delete;
    TemporaryFixture & operator=(const TemporaryFixture &) = delete;

    ~TemporaryFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    [[nodiscard]]
    const std::filesystem::path & directory() const noexcept
    {
        return directory_;
    }

    [[nodiscard]]
    const std::filesystem::path & file() const noexcept
    {
        return file_;
    }

private:
    std::filesystem::path directory_;
    std::filesystem::path file_;
};

Filesystem make_filesystem()
{
#if defined(_WIN32)
    auto result = Filesystem::create(std::make_unique<WindowsFilesystemBackend>());
#else
    auto result = Filesystem::create(std::make_unique<PosixFilesystemBackend>());
#endif
    assert(result.has_value());
    return std::move(*result);
}

void assert_error_code(const FilesystemError & error, FilesystemErrorCode expected)
{
    assert(error.category() == common::ErrorCategory::Filesystem);
    assert(error.code() == std::to_underlying(expected));
}

void test_invalid_and_moved_from_objects(const TemporaryFixture & fixture)
{
    auto invalid_filesystem = Filesystem::create(nullptr);
    assert(!invalid_filesystem.has_value());
    assert_error_code(invalid_filesystem.error(), FilesystemErrorCode::InvalidState);

    auto invalid_handle = FileHandle::create(nullptr);
    assert(!invalid_handle.has_value());
    assert_error_code(invalid_handle.error(), FilesystemErrorCode::InvalidState);

    auto filesystem = make_filesystem();
    auto moved_filesystem = std::move(filesystem);
    auto moved_from_exists = filesystem.exists(fixture.file());
    assert(!moved_from_exists.has_value());
    assert_error_code(moved_from_exists.error(), FilesystemErrorCode::InvalidState);

    auto handle_result = moved_filesystem.open(fixture.file());
    assert(handle_result.has_value());
    auto handle = std::move(*handle_result);
    auto moved_handle = std::move(handle);
    auto moved_from_size = handle.size();
    assert(!moved_from_size.has_value());
    assert_error_code(moved_from_size.error(), FilesystemErrorCode::InvalidState);
    assert(moved_handle.close().has_value());
}

void test_exists_and_open_errors(const TemporaryFixture & fixture)
{
    auto filesystem = make_filesystem();

    auto file_exists = filesystem.exists(fixture.file());
    assert(file_exists.has_value() && *file_exists);

    auto directory_exists = filesystem.exists(fixture.directory());
    assert(directory_exists.has_value() && *directory_exists);

    const auto missing_path = fixture.directory() / "missing.bin";
    auto missing_exists = filesystem.exists(missing_path);
    assert(missing_exists.has_value() && !*missing_exists);

    auto missing_handle = filesystem.open(missing_path);
    assert(!missing_handle.has_value());
    assert_error_code(missing_handle.error(), FilesystemErrorCode::NotFound);

    auto directory_handle = filesystem.open(fixture.directory());
    assert(!directory_handle.has_value());
    assert_error_code(directory_handle.error(), FilesystemErrorCode::NotAFile);

    auto empty_path_handle = filesystem.open({});
    assert(!empty_path_handle.has_value());
    assert_error_code(empty_path_handle.error(), FilesystemErrorCode::InvalidPath);

    FileOpenOptions invalid_options;
    invalid_options.access = static_cast<FileAccess>(255);
    auto invalid_options_handle = filesystem.open(fixture.file(), invalid_options);
    assert(!invalid_options_handle.has_value());
    assert_error_code(invalid_options_handle.error(), FilesystemErrorCode::InvalidArgument);
}

void test_read_size_and_close(const TemporaryFixture & fixture)
{
    auto filesystem = make_filesystem();
    auto handle_result = filesystem.open(fixture.file());
    assert(handle_result.has_value());
    auto handle = std::move(*handle_result);

    auto size = handle.size();
    assert(size.has_value() && *size == FILE_CONTENT.size());

    std::array<std::byte, 16> buffer {};
    auto read = handle.read_at(4, buffer);
    assert(read.has_value() && *read == 5);
    const std::string_view content(reinterpret_cast<const char *>(buffer.data()), *read);
    assert(content == "Infer");

    auto eof_read = handle.read_at(FILE_CONTENT.size(), buffer);
    assert(eof_read.has_value() && *eof_read == 0);

    auto invalid_offset_read = handle.read_at(UINT64_MAX, std::span<std::byte>(buffer).first(1));
    assert(!invalid_offset_read.has_value());
    assert_error_code(invalid_offset_read.error(), FilesystemErrorCode::InvalidArgument);

    assert(handle.close().has_value());
    assert(handle.close().has_value());

    auto closed_read = handle.read_at(0, buffer);
    assert(!closed_read.has_value());
    assert_error_code(closed_read.error(), FilesystemErrorCode::ClosedHandle);

    auto closed_size = handle.size();
    assert(!closed_size.has_value());
    assert_error_code(closed_size.error(), FilesystemErrorCode::ClosedHandle);
}

} // namespace

int main()
{
    const TemporaryFixture fixture;
    test_invalid_and_moved_from_objects(fixture);
    test_exists_and_open_errors(fixture);
    test_read_size_and_close(fixture);
}
