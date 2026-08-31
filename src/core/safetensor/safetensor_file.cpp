#include "core/safetensor/safetensor_file.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <utility>

#include "core/io/binary_reader.hpp"
#include "core/io/file_byte_reader.hpp"
#include "core/io/io_error.hpp"
#include "core/json/json.hpp"

namespace liteinfer::core::safetensor
{

namespace
{

constexpr std::uint64_t HEADER_SIZE_LIMIT = 100'000'000;
constexpr std::uint64_t HEADER_PREFIX_SIZE = sizeof(std::uint64_t);
constexpr std::string_view METADATA_KEY = "__metadata__";

std::expected<std::uint64_t, SafetensorError> parse_unsigned_integer(
    const json::Document & value,
    SafetensorErrorCode code,
    std::string_view message
)
{
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }

    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value >= 0) {
            return static_cast<std::uint64_t>(signed_value);
        }
    }

    return std::unexpected(SafetensorError {code, message});
}

std::expected<std::uint64_t, SafetensorError> parse_shape_dimension(const json::Document & value)
{
    return parse_unsigned_integer(
        value,
        SafetensorErrorCode::InvalidShape,
        "Safetensors shape dimension must be a non-negative integer"
    );
}

std::expected<std::uint64_t, SafetensorError> parse_offset(const json::Document & value)
{
    return parse_unsigned_integer(
        value,
        SafetensorErrorCode::InvalidOffsets,
        "Safetensors data offset must be a non-negative integer"
    );
}

std::optional<std::uint64_t> dtype_element_size(std::string_view dtype)
{
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" ||
        dtype == "F8_E5M2" || dtype == "F8_E8M0" || dtype == "F8_E4M3FNUZ" ||
        dtype == "F8_E5M2FNUZ") {
        return 1;
    }
    if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") {
        return 2;
    }
    if (dtype == "U32" || dtype == "I32" || dtype == "F32") {
        return 4;
    }
    if (dtype == "U64" || dtype == "I64" || dtype == "F64" || dtype == "C64") {
        return 8;
    }
    return std::nullopt;
}

std::expected<std::uint64_t, SafetensorError> shape_numel(const std::vector<std::uint64_t> & shape)
{
    std::uint64_t numel = 1;
    for (const auto dimension : shape) {
        if (dimension != 0 && numel > std::numeric_limits<std::uint64_t>::max() / dimension)
            [[unlikely]] {
            return std::unexpected(SafetensorError {
                SafetensorErrorCode::InvalidShape,
                "Safetensors shape size overflows"
            });
        }
        numel *= dimension;
    }
    return numel;
}

struct ParsedHeader
{
    std::map<std::string, TensorDescriptor> tensors;
    std::map<std::string, std::string> metadata;
};

struct DataRange
{
    std::uint64_t begin;
    std::uint64_t end;
    std::string name;
};

std::expected<TensorDescriptor, SafetensorError>
parse_tensor_descriptor(const json::Document & value, std::uint64_t data_region_size)
{
    if (!value.is_object()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidTensorInfo,
            "Tensor entry must be a JSON object"
        });
    }

    if (!value.contains("dtype") || !value["dtype"].is_string()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidDtype,
            "Tensor entry must contain a string dtype"
        });
    }
    const auto dtype = value["dtype"].get<std::string>();
    if (dtype.empty()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidDtype,
            "Tensor dtype must not be empty"
        });
    }

    if (!value.contains("shape") || !value["shape"].is_array()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidShape,
            "Tensor entry must contain a shape array"
        });
    }
    std::vector<std::uint64_t> shape;
    shape.reserve(value["shape"].size());
    for (const auto & dimension : value["shape"]) {
        auto parsed_dimension = parse_shape_dimension(dimension);
        if (!parsed_dimension) [[unlikely]] {
            return std::unexpected(std::move(parsed_dimension.error()));
        }
        shape.push_back(*parsed_dimension);
    }
    auto numel = shape_numel(shape);
    if (!numel) [[unlikely]] {
        return std::unexpected(std::move(numel.error()));
    }

    if (!value.contains("data_offsets") || !value["data_offsets"].is_array() ||
        value["data_offsets"].size() != 2) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidOffsets,
            "Tensor entry must contain exactly two data offsets"
        });
    }
    auto data_begin = parse_offset(value["data_offsets"][0]);
    if (!data_begin) [[unlikely]] {
        return std::unexpected(std::move(data_begin.error()));
    }
    auto data_end = parse_offset(value["data_offsets"][1]);
    if (!data_end) [[unlikely]] {
        return std::unexpected(std::move(data_end.error()));
    }
    if (*data_begin > *data_end || *data_end > data_region_size) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidOffsets,
            "Tensor data offsets are out of range"
        });
    }

    if (const auto element_size = dtype_element_size(dtype); element_size.has_value()) {
        if (*numel > std::numeric_limits<std::uint64_t>::max() / *element_size) [[unlikely]] {
            return std::unexpected(SafetensorError {
                SafetensorErrorCode::TensorSizeMismatch,
                "Tensor byte size overflows"
            });
        }
        const auto expected_bytes = *numel * *element_size;
        if (*data_end - *data_begin != expected_bytes) [[unlikely]] {
            return std::unexpected(SafetensorError {
                SafetensorErrorCode::TensorSizeMismatch,
                "Tensor byte size does not match shape"
            });
        }
    }

    return TensorDescriptor {dtype, std::move(shape), *data_begin, *data_end};
}

std::expected<ParsedHeader, SafetensorError>
parse_header(const json::Document & document, std::uint64_t data_region_size)
{
    if (!document.is_object()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidRoot,
            "Safetensors header root must be an object"
        });
    }

    ParsedHeader parsed;
    std::vector<DataRange> ranges;
    ranges.reserve(document.size());

    for (auto iterator = document.begin(); iterator != document.end(); ++iterator) {
        if (iterator.key() == METADATA_KEY) {
            if (!iterator.value().is_object()) [[unlikely]] {
                return std::unexpected(SafetensorError {
                    SafetensorErrorCode::InvalidMetadata,
                    "Safetensors metadata must be an object"
                });
            }
            for (auto metadata_iterator = iterator.value().begin();
                 metadata_iterator != iterator.value().end();
                 ++metadata_iterator) {
                if (!metadata_iterator.value().is_string()) [[unlikely]] {
                    return std::unexpected(SafetensorError {
                        SafetensorErrorCode::InvalidMetadata,
                        "Safetensors metadata values must be strings"
                    });
                }
                parsed.metadata.emplace(
                    metadata_iterator.key(),
                    metadata_iterator.value().get<std::string>()
                );
            }
            continue;
        }

        auto descriptor = parse_tensor_descriptor(iterator.value(), data_region_size);
        if (!descriptor) [[unlikely]] {
            return std::unexpected(std::move(descriptor.error()));
        }
        ranges.push_back({descriptor->data_begin, descriptor->data_end, iterator.key()});
        parsed.tensors.emplace(iterator.key(), std::move(*descriptor));
    }

    std::sort(ranges.begin(), ranges.end(), [](const DataRange & left, const DataRange & right) {
        if (left.begin != right.begin) {
            return left.begin < right.begin;
        }
        return left.end < right.end;
    });
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].begin < ranges[index - 1].end) [[unlikely]] {
            return std::unexpected(SafetensorError {
                SafetensorErrorCode::InvalidOffsets,
                "Tensor data ranges overlap"
            });
        }
    }

    return parsed;
}

} // namespace

SafetensorFile::SafetensorFile(
    filesystem::FileHandle file,
    std::uint64_t file_size,
    std::uint64_t data_region_begin,
    std::map<std::string, TensorDescriptor> tensors,
    std::map<std::string, std::string> metadata
) noexcept
    : file_(std::move(file))
    , file_size_(file_size)
    , data_region_begin_(data_region_begin)
    , tensors_(std::move(tensors))
    , metadata_(std::move(metadata))
{}

SafetensorFile::SafetensorFile(SafetensorFile &&) noexcept = default;

SafetensorFile & SafetensorFile::operator=(SafetensorFile &&) noexcept = default;

SafetensorFile::~SafetensorFile() = default;

std::expected<SafetensorFile, SafetensorError>
SafetensorFile::open(filesystem::Filesystem & filesystem, const std::filesystem::path & path)
{
    auto file_result = filesystem.open(path);
    if (!file_result) [[unlikely]] {
        return std::unexpected(std::move(file_result.error()));
    }
    auto file = std::move(*file_result);

    auto file_size_result = file.size();
    if (!file_size_result) [[unlikely]] {
        return std::unexpected(std::move(file_size_result.error()));
    }
    const auto file_size = *file_size_result;
    if (file_size < HEADER_PREFIX_SIZE) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::HeaderTooSmall,
            "Safetensors file is smaller than its header prefix"
        });
    }

    io::FileByteReader byte_reader(file);
    io::LittleEndianBinaryReader binary_reader(byte_reader, io::BinaryDecodeLimits {file_size, 0});
    auto header_size_result = binary_reader.read_u64();
    if (!header_size_result) [[unlikely]] {
        return std::unexpected(std::move(header_size_result.error()));
    }
    const auto header_size = *header_size_result;
    if (header_size > HEADER_SIZE_LIMIT) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::HeaderTooLarge,
            "Safetensors header exceeds the size limit"
        });
    }
    if (header_size > std::numeric_limits<std::size_t>::max() ||
        header_size > std::numeric_limits<std::uint64_t>::max() - HEADER_PREFIX_SIZE)
        [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidHeaderLength,
            "Safetensors header length overflows"
        });
    }

    const auto data_region_begin = HEADER_PREFIX_SIZE + header_size;
    if (data_region_begin > file_size) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidHeaderLength,
            "Safetensors header exceeds the file size"
        });
    }

    std::vector<std::byte> header_bytes(static_cast<std::size_t>(header_size));
    auto header_read = byte_reader.read_exact(header_bytes);
    if (!header_read) [[unlikely]] {
        return std::unexpected(std::move(header_read.error()));
    }

    if (header_bytes.empty() || std::to_integer<unsigned char>(header_bytes.front()) != '{')
        [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidJson,
            "Safetensors header must begin with an object"
        });
    }

    const std::string header_text(
        reinterpret_cast<const char *>(header_bytes.data()),
        header_bytes.size()
    );
    json::Document document;
    try {
        document = json::Document::parse(header_text);
    } catch (const nlohmann::json::exception &) {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidJson,
            "Safetensors header is not valid JSON"
        });
    }

    const auto data_region_size = file_size - data_region_begin;
    auto parsed_header = parse_header(document, data_region_size);
    if (!parsed_header) [[unlikely]] {
        return std::unexpected(std::move(parsed_header.error()));
    }

    return SafetensorFile(
        std::move(file),
        file_size,
        data_region_begin,
        std::move(parsed_header->tensors),
        std::move(parsed_header->metadata)
    );
}

std::vector<std::string> SafetensorFile::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto & entry : tensors_) {
        names.push_back(entry.first);
    }
    return names;
}

std::expected<TensorDescriptor, SafetensorError> SafetensorFile::tensor_info(
    std::string_view name
) const
{
    const auto iterator = tensors_.find(std::string(name));
    if (iterator == tensors_.end()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::TensorNotFound,
            "Safetensors tensor name was not found"
        });
    }
    return iterator->second;
}

const std::map<std::string, std::string> & SafetensorFile::metadata() const noexcept
{
    return metadata_;
}

std::expected<std::vector<std::byte>, SafetensorError> SafetensorFile::read_tensor(
    std::string_view name
)
{
    const auto iterator = tensors_.find(std::string(name));
    if (iterator == tensors_.end()) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::TensorNotFound,
            "Safetensors tensor name was not found"
        });
    }

    const auto & descriptor = iterator->second;
    const auto byte_count = descriptor.data_end - descriptor.data_begin;
    if (byte_count > std::numeric_limits<std::size_t>::max() ||
        descriptor.data_begin > std::numeric_limits<std::uint64_t>::max() - data_region_begin_)
        [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidOffsets,
            "Safetensors tensor offset cannot be represented"
        });
    }
    const auto absolute_offset = data_region_begin_ + descriptor.data_begin;
    if (byte_count > std::numeric_limits<std::uint64_t>::max() - absolute_offset ||
        absolute_offset + byte_count > file_size_) [[unlikely]] {
        return std::unexpected(SafetensorError {
            SafetensorErrorCode::InvalidOffsets,
            "Safetensors tensor range exceeds the file size"
        });
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(byte_count));
    std::size_t total_read = 0;
    while (total_read < bytes.size()) {
        auto read_result = file_.read_at(
            absolute_offset + static_cast<std::uint64_t>(total_read),
            std::span<std::byte>(bytes).subspan(total_read)
        );
        if (!read_result) [[unlikely]] {
            return std::unexpected(std::move(read_result.error()));
        }
        if (*read_result == 0) [[unlikely]] {
            return std::unexpected(io::IoError {
                io::IoErrorCode::UnexpectedEof,
                "Tensor data ended before the requested range"
            });
        }
        if (*read_result > bytes.size() - total_read) [[unlikely]] {
            return std::unexpected(io::IoError {
                io::IoErrorCode::InvalidData,
                "File handle returned too many tensor bytes"
            });
        }
        total_read += *read_result;
    }
    return bytes;
}

} // namespace liteinfer::core::safetensor
