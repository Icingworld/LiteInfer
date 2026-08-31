#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

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
#include <vector>

#include "core/common/error.hpp"
#include "core/filesystem/backend/file_handle_backend.hpp"
#include "core/filesystem/backend/filesystem_backend.hpp"
#if defined(_WIN32)
#include "core/filesystem/backend/windows/windows_filesystem_backend.hpp"
#else
#include "core/filesystem/backend/posix/posix_filesystem_backend.hpp"
#endif
#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/filesystem/filesystem_error.hpp"
#include "core/safetensor/safetensor_error.hpp"
#include "core/safetensor/safetensor_file.hpp"

namespace
{

using namespace liteinfer::core;
using filesystem::FileHandle;
using filesystem::Filesystem;
using filesystem::FilesystemError;
using filesystem::FilesystemErrorCode;
using safetensor::SafetensorError;
using safetensor::SafetensorErrorCode;
using safetensor::SafetensorFile;

#if defined(_WIN32)
using NativeFilesystemBackend = filesystem::backend::windows::WindowsFilesystemBackend;
#else
using NativeFilesystemBackend = filesystem::backend::posix::PosixFilesystemBackend;
#endif

void append_u64_le(std::vector<std::byte> & bytes, std::uint64_t value)
{
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        bytes.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
    }
}

std::vector<std::byte>
make_file_bytes(std::string_view header, const std::vector<std::byte> & data = {})
{
    std::vector<std::byte> bytes;
    bytes.reserve(sizeof(std::uint64_t) + header.size() + data.size());
    append_u64_le(bytes, header.size());
    for (const auto character : header) {
        bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    }
    bytes.insert(bytes.end(), data.begin(), data.end());
    return bytes;
}

class TemporaryFixture
{
public:
    TemporaryFixture()
        : directory_(
              std::filesystem::temp_directory_path() /
              ("liteinfer_safetensor_test_" +
#if defined(_WIN32)
               std::to_string(static_cast<unsigned long long>(::_getpid()))
#else
               std::to_string(static_cast<unsigned long long>(::getpid()))
#endif
              )
          )
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        assert(!error);
        assert(std::filesystem::create_directory(directory_));
    }

    TemporaryFixture(const TemporaryFixture &) = delete;
    TemporaryFixture & operator=(const TemporaryFixture &) = delete;

    ~TemporaryFixture()
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    [[nodiscard]]
    std::filesystem::path write_bytes(const std::vector<std::byte> & bytes)
    {
        const auto path = directory_ / ("case_" + std::to_string(next_file_++) + ".safetensors");
        std::ofstream stream(path, std::ios::binary);
        assert(stream.is_open());
        if (!bytes.empty()) {
            stream.write(
                reinterpret_cast<const char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size())
            );
        }
        assert(stream.good());
        return path;
    }

    [[nodiscard]]
    std::filesystem::path write(std::string_view header, const std::vector<std::byte> & data = {})
    {
        return write_bytes(make_file_bytes(header, data));
    }

    [[nodiscard]]
    const std::filesystem::path & directory() const noexcept
    {
        return directory_;
    }

private:
    std::filesystem::path directory_;
    std::size_t next_file_ {0};
};

Filesystem make_filesystem()
{
    auto result = Filesystem::create(std::make_unique<NativeFilesystemBackend>());
    assert(result.has_value());
    return std::move(*result);
}

std::expected<SafetensorFile, SafetensorError> open_file(const std::filesystem::path & path)
{
    auto filesystem = make_filesystem();
    return SafetensorFile::open(filesystem, path);
}

void assert_safetensor_error(const SafetensorError & error, SafetensorErrorCode expected)
{
    assert(error.category() == common::ErrorCategory::Safetensor);
    assert(error.code() == std::to_underlying(expected));
}

void assert_filesystem_error(const common::Error & error, FilesystemErrorCode expected)
{
    assert(error.category() == common::ErrorCategory::Filesystem);
    assert(error.code() == std::to_underlying(expected));
}

void test_valid_file_and_data_access()
{
    TemporaryFixture fixture;
    const std::vector data {
        std::byte {0x00},
        std::byte {0x00},
        std::byte {0x80},
        std::byte {0x3f},
        std::byte {0x00},
        std::byte {0x00},
        std::byte {0x00},
        std::byte {0x40},
    };
    const auto path = fixture.write(
        R"({"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"format":"test"}})",
        data
    );

    auto result = open_file(path);
    assert(result.has_value());
    auto file = std::move(*result);

    const auto names = file.tensor_names();
    assert(names.size() == 1);
    assert(names[0] == "weight");

    auto descriptor = file.tensor_info("weight");
    assert(descriptor.has_value());
    assert(descriptor->dtype == "F32");
    assert(descriptor->shape == std::vector<std::uint64_t> {2});
    assert(descriptor->data_begin == 0);
    assert(descriptor->data_end == data.size());

    assert(file.metadata().size() == 1);
    assert(file.metadata().at("format") == "test");

    auto tensor_bytes = file.read_tensor("weight");
    assert(tensor_bytes.has_value());
    assert(*tensor_bytes == data);
}

void test_multiple_tensors_empty_tensor_and_unknown_dtype()
{
    TemporaryFixture fixture;
    const std::vector data {
        std::byte {0x01},
        std::byte {0x02},
        std::byte {0x03},
        std::byte {0x04},
        std::byte {0x05},
        std::byte {0x06},
        std::byte {0x07},
    };
    const auto path = fixture.write(
        R"({"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]},"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]},"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[7,7]}}   )",
        data
    );

    auto result = open_file(path);
    assert(result.has_value());
    auto file = std::move(*result);

    const auto names = file.tensor_names();
    const std::vector<std::string> expected_names {"a", "b", "empty"};
    assert(names == expected_names);

    auto unknown_descriptor = file.tensor_info("a");
    assert(unknown_descriptor.has_value());
    assert(unknown_descriptor->dtype == "F4");

    auto empty_bytes = file.read_tensor("empty");
    assert(empty_bytes.has_value());
    assert(empty_bytes->empty());
}

void test_header_errors()
{
    TemporaryFixture fixture;

    const auto too_small = fixture.write_bytes({std::byte {0x01}, std::byte {0x02}});
    auto too_small_result = open_file(too_small);
    assert(!too_small_result.has_value());
    assert_safetensor_error(too_small_result.error(), SafetensorErrorCode::HeaderTooSmall);

    std::vector<std::byte> too_large_bytes;
    append_u64_le(too_large_bytes, 100'000'001);
    const auto too_large = fixture.write_bytes(too_large_bytes);
    auto too_large_result = open_file(too_large);
    assert(!too_large_result.has_value());
    assert_safetensor_error(too_large_result.error(), SafetensorErrorCode::HeaderTooLarge);

    std::vector<std::byte> invalid_length_bytes;
    append_u64_le(invalid_length_bytes, 10);
    const auto invalid_length = fixture.write_bytes(invalid_length_bytes);
    auto invalid_length_result = open_file(invalid_length);
    assert(!invalid_length_result.has_value());
    assert_safetensor_error(
        invalid_length_result.error(),
        SafetensorErrorCode::InvalidHeaderLength
    );

    const auto invalid_json = fixture.write(R"({"tensor":)");
    auto invalid_json_result = open_file(invalid_json);
    assert(!invalid_json_result.has_value());
    assert_safetensor_error(invalid_json_result.error(), SafetensorErrorCode::InvalidJson);

    const auto non_object = fixture.write("[]");
    auto non_object_result = open_file(non_object);
    assert(!non_object_result.has_value());
    assert_safetensor_error(non_object_result.error(), SafetensorErrorCode::InvalidJson);
}

void test_tensor_and_metadata_validation()
{
    TemporaryFixture fixture;

    const auto missing_dtype = fixture.write(R"({"tensor":{"shape":[1],"data_offsets":[0,4]}})");
    auto missing_dtype_result = open_file(missing_dtype);
    assert(!missing_dtype_result.has_value());
    assert_safetensor_error(missing_dtype_result.error(), SafetensorErrorCode::InvalidDtype);

    const auto wrong_shape =
        fixture.write(R"({"tensor":{"dtype":"F32","shape":1,"data_offsets":[0,4]}})");
    auto wrong_shape_result = open_file(wrong_shape);
    assert(!wrong_shape_result.has_value());
    assert_safetensor_error(wrong_shape_result.error(), SafetensorErrorCode::InvalidShape);

    const auto missing_offsets = fixture.write(R"({"tensor":{"dtype":"F32","shape":[1]}})");
    auto missing_offsets_result = open_file(missing_offsets);
    assert(!missing_offsets_result.has_value());
    assert_safetensor_error(missing_offsets_result.error(), SafetensorErrorCode::InvalidOffsets);

    const auto invalid_metadata = fixture.write(R"({"__metadata__":{"format":1}})");
    auto invalid_metadata_result = open_file(invalid_metadata);
    assert(!invalid_metadata_result.has_value());
    assert_safetensor_error(invalid_metadata_result.error(), SafetensorErrorCode::InvalidMetadata);

    const auto overflowing_shape = fixture.write(
        R"({"tensor":{"dtype":"CUSTOM","shape":[18446744073709551615,2],"data_offsets":[0,0]}})"
    );
    auto overflowing_shape_result = open_file(overflowing_shape);
    assert(!overflowing_shape_result.has_value());
    assert_safetensor_error(overflowing_shape_result.error(), SafetensorErrorCode::InvalidShape);

    const auto reversed_offsets =
        fixture.write(R"({"tensor":{"dtype":"CUSTOM","shape":[1],"data_offsets":[2,1]}})");
    auto reversed_offsets_result = open_file(reversed_offsets);
    assert(!reversed_offsets_result.has_value());
    assert_safetensor_error(reversed_offsets_result.error(), SafetensorErrorCode::InvalidOffsets);

    const auto out_of_range =
        fixture.write(R"({"tensor":{"dtype":"CUSTOM","shape":[1],"data_offsets":[0,1]}})");
    auto out_of_range_result = open_file(out_of_range);
    assert(!out_of_range_result.has_value());
    assert_safetensor_error(out_of_range_result.error(), SafetensorErrorCode::InvalidOffsets);

    const std::vector overlap_data(6, std::byte {0});
    const auto overlapping = fixture.write(
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})",
        overlap_data
    );
    auto overlapping_result = open_file(overlapping);
    assert(!overlapping_result.has_value());
    assert_safetensor_error(overlapping_result.error(), SafetensorErrorCode::InvalidOffsets);

    const std::vector mismatch_data(4, std::byte {0});
    const auto mismatched_size = fixture.write(
        R"({"tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
        mismatch_data
    );
    auto mismatched_size_result = open_file(mismatched_size);
    assert(!mismatched_size_result.has_value());
    assert_safetensor_error(
        mismatched_size_result.error(),
        SafetensorErrorCode::TensorSizeMismatch
    );
}

void test_missing_tensor_and_filesystem_error()
{
    TemporaryFixture fixture;
    const auto path = fixture.write(
        R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
        std::vector(4, std::byte {0})
    );
    auto result = open_file(path);
    assert(result.has_value());
    auto file = std::move(*result);

    auto missing_info = file.tensor_info("missing");
    assert(!missing_info.has_value());
    assert_safetensor_error(missing_info.error(), SafetensorErrorCode::TensorNotFound);

    auto missing_bytes = file.read_tensor("missing");
    assert(!missing_bytes.has_value());
    assert_safetensor_error(missing_bytes.error(), SafetensorErrorCode::TensorNotFound);

    const auto missing_path = fixture.directory() / "missing.safetensors";
    auto missing_file = open_file(missing_path);
    assert(!missing_file.has_value());
    assert_filesystem_error(missing_file.error(), FilesystemErrorCode::NotFound);
}

class FailingFileHandleBackend final : public filesystem::backend::FileHandleBackend
{
public:
    std::expected<void, FilesystemError> close() override
    {
        return {};
    }

    std::expected<std::size_t, FilesystemError>
    read_at(std::uint64_t, std::span<std::byte>) override
    {
        return std::unexpected(
            FilesystemError {FilesystemErrorCode::PermissionDenied, "synthetic read failure"}
        );
    }

    std::expected<std::uint64_t, FilesystemError> size() override
    {
        return 8;
    }
};

class FailingFilesystemBackend final : public filesystem::backend::FilesystemBackend
{
public:
    std::expected<FileHandle, FilesystemError>
    open(const std::filesystem::path &, const filesystem::FileOpenOptions &) override
    {
        return FileHandle::create(std::make_unique<FailingFileHandleBackend>());
    }

    std::expected<bool, FilesystemError> exists(const std::filesystem::path &) override
    {
        return false;
    }
};

void test_read_error_is_preserved()
{
    auto filesystem_result = Filesystem::create(std::make_unique<FailingFilesystemBackend>());
    assert(filesystem_result.has_value());
    auto filesystem = std::move(*filesystem_result);

    auto result = SafetensorFile::open(filesystem, "synthetic.safetensors");
    assert(!result.has_value());
    assert_filesystem_error(result.error(), FilesystemErrorCode::PermissionDenied);
}

} // namespace

int main()
{
    test_valid_file_and_data_access();
    test_multiple_tensors_empty_tensor_and_unknown_dtype();
    test_header_errors();
    test_tensor_and_metadata_validation();
    test_missing_tensor_and_filesystem_error();
    test_read_error_is_preserved();
}
