#include "core/tensor/tensor_old.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace liteinfer::core::tensor
{

namespace
{

struct TensorLayout
{
    std::size_t numel;
    std::size_t element_size;
    std::size_t bytes;
};

// 获取数据类型对应每个元素的字节大小
[[nodiscard]]
constexpr std::size_t element_size_of(common::data_type::DataType data_type) noexcept
{
    switch (data_type) {
    case common::data_type::DataType::Float16:
        [[fallthrough]];
    case common::data_type::DataType::BFloat16:
        return 2;
    case common::data_type::DataType::Float32:
        return 4;
    case common::data_type::DataType::Int8:
        [[fallthrough]];
    case common::data_type::DataType::Bool:
        return 1;
    case common::data_type::DataType::Int32:
        return 4;
    case common::data_type::DataType::Int64:
        return 8;
    default:
        return 0;
    }
}

// 获取数据类型对应每个元素的字节大小
[[nodiscard]]
std::expected<std::size_t, TensorError> element_size(common::data_type::DataType data_type) noexcept
{
    const std::size_t size = element_size_of(data_type);
    if (size == 0) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidDataType, "Unsupported tensor data type")
        );
    }

    return size;
}

// 计算 Tensor 的元素个数、元素字节大小与总字节大小
[[nodiscard]]
std::expected<TensorLayout, TensorError>
compute_layout(common::data_type::DataType data_type, const Shape & shape) noexcept
{
    auto numel = shape.numel();
    if (!numel) [[unlikely]] {
        return std::unexpected(std::move(numel).error());
    }

    auto element_size_result = element_size(data_type);
    if (!element_size_result) [[unlikely]] {
        return std::unexpected(std::move(element_size_result).error());
    }

    const std::size_t count = *numel;
    const std::size_t element_bytes = *element_size_result;
    if (count != 0 && element_bytes > std::numeric_limits<std::size_t>::max() / count)
        [[unlikely]] {
        return std::unexpected(TensorError(TensorErrorCode::Overflow, "Tensor byte size overflow"));
    }

    return TensorLayout {
        .numel = count,
        .element_size = element_bytes,
        .bytes = count * element_bytes,
    };
}

[[nodiscard]]
Strides contiguous_strides_or_throw(const Shape & shape)
{
    auto result = Strides::row_major(shape);
    if (!result) [[unlikely]] {
        throw std::invalid_argument(std::string(result.error().message()));
    }
    return std::move(*result);
}

[[nodiscard]]
std::expected<std::size_t, TensorError> compute_storage_span_bytes(
    const Shape & shape,
    const Strides & strides,
    std::size_t element_size
) noexcept
{
    const auto dimensions = shape.values();
    const auto stride_values = strides.values();
    if (dimensions.size() != stride_values.size()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::ShapeMismatch, "Tensor shape and strides rank mismatch")
        );
    }

    for (const auto dimension : dimensions) {
        if (dimension == 0) {
            return std::size_t {0};
        }
    }

    std::size_t maximum_element_offset = 0;
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        const std::size_t extent = dimensions[index] - 1;
        if (extent != 0 && stride_values[index] > std::numeric_limits<std::size_t>::max() / extent)
            [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::Overflow, "Tensor view offset overflow")
            );
        }

        const std::size_t dimension_offset = extent * stride_values[index];
        if (maximum_element_offset > std::numeric_limits<std::size_t>::max() - dimension_offset)
            [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::Overflow, "Tensor view offset overflow")
            );
        }
        maximum_element_offset += dimension_offset;
    }

    if (maximum_element_offset == std::numeric_limits<std::size_t>::max()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::Overflow, "Tensor view byte span overflow")
        );
    }
    const std::size_t element_span = maximum_element_offset + 1;
    if (element_size == 0 || element_span > std::numeric_limits<std::size_t>::max() / element_size)
        [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::Overflow, "Tensor view byte span overflow")
        );
    }

    return element_span * element_size;
}

} // namespace

Tensor::Tensor(
    common::data_type::DataType data_type,
    Shape shape,
    Strides strides,
    std::shared_ptr<std::vector<std::byte>> storage,
    std::size_t data_offset_bytes,
    std::size_t data_span_bytes,
    std::size_t numel,
    std::size_t element_size
)
    : data_type_(data_type)
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , storage_(std::move(storage))
    , data_offset_bytes_(data_offset_bytes)
    , data_span_bytes_(data_span_bytes)
    , numel_(numel)
    , element_size_(element_size)
{}

Tensor::Tensor(const Tensor & other)
    : data_type_(other.data_type_)
    , shape_(other.shape_)
    , strides_(contiguous_strides_or_throw(other.shape_))
    , storage_(std::make_shared<std::vector<std::byte>>(other.bytes(), std::byte {0}))
    , data_offset_bytes_(0)
    , data_span_bytes_(other.bytes())
    , numel_(other.numel_)
    , element_size_(other.element_size_)
{
    auto result = copy_from(other);
    if (!result) [[unlikely]] {
        throw std::invalid_argument(std::string(result.error().message()));
    }
}

Tensor & Tensor::operator=(const Tensor & other)
{
    if (this == &other) {
        return *this;
    }

    Tensor copy(other);
    *this = std::move(copy);
    return *this;
}

std::expected<Tensor, TensorError>
Tensor::allocate(common::data_type::DataType data_type, Shape shape)
{
    auto layout = compute_layout(data_type, shape);
    if (!layout) [[unlikely]] {
        return std::unexpected(std::move(layout).error());
    }

    auto strides = Strides::row_major(shape);
    if (!strides) [[unlikely]] {
        return std::unexpected(std::move(strides).error());
    }

    auto storage = std::make_shared<std::vector<std::byte>>(layout->bytes, std::byte {0});
    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(storage),
        0,
        layout->bytes,
        layout->numel,
        layout->element_size
    );
}

std::expected<Tensor, TensorError> Tensor::from_bytes(
    common::data_type::DataType data_type,
    Shape shape,
    std::span<const std::byte> bytes
)
{
    auto layout = compute_layout(data_type, shape);
    if (!layout) [[unlikely]] {
        return std::unexpected(std::move(layout).error());
    }

    if (bytes.size() != layout->bytes) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidByteSize, "Tensor byte size mismatch")
        );
    }

    auto strides = Strides::row_major(shape);
    if (!strides) [[unlikely]] {
        return std::unexpected(std::move(strides).error());
    }

    auto storage = std::make_shared<std::vector<std::byte>>(bytes.begin(), bytes.end());
    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(storage),
        0,
        layout->bytes,
        layout->numel,
        layout->element_size
    );
}

std::expected<Tensor, TensorError>
Tensor::narrow(std::size_t dimension, std::size_t start, std::size_t length) const
{
    const auto dimensions = shape_.values();
    if (dimension >= dimensions.size()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::IndexOutOfRange, "Tensor dimension out of range")
        );
    }

    if (start > dimensions[dimension] || length > dimensions[dimension] - start) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::IndexOutOfRange, "Tensor narrow range out of range")
        );
    }

    auto narrowed_dimensions = std::vector<std::size_t>(dimensions.begin(), dimensions.end());
    narrowed_dimensions[dimension] = length;
    Shape narrowed_shape(std::move(narrowed_dimensions));
    auto narrowed_numel = narrowed_shape.numel();
    if (!narrowed_numel) [[unlikely]] {
        return std::unexpected(std::move(narrowed_numel).error());
    }

    const auto stride_values = strides_.values();
    if (stride_values.size() != dimensions.size() || element_size_ == 0) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::ShapeMismatch, "Tensor layout is not initialized")
        );
    }
    if (start != 0 && stride_values[dimension] > std::numeric_limits<std::size_t>::max() / start)
        [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::Overflow, "Tensor narrow offset overflow")
        );
    }
    const std::size_t offset_elements = start * stride_values[dimension];
    if (offset_elements > std::numeric_limits<std::size_t>::max() / element_size_) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::Overflow, "Tensor narrow byte offset overflow")
        );
    }
    const std::size_t offset_element_bytes = offset_elements * element_size_;
    if (data_offset_bytes_ > std::numeric_limits<std::size_t>::max() - offset_element_bytes)
        [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::Overflow, "Tensor narrow byte offset overflow")
        );
    }
    const std::size_t offset_bytes = data_offset_bytes_ + offset_element_bytes;

    auto data_span = compute_storage_span_bytes(narrowed_shape, strides_, element_size_);
    if (!data_span) [[unlikely]] {
        return std::unexpected(std::move(data_span).error());
    }
    if (!storage_ || offset_bytes > storage_->size() ||
        *data_span > storage_->size() - offset_bytes) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::IndexOutOfRange, "Tensor narrow range exceeds storage")
        );
    }

    return Tensor(
        data_type_,
        std::move(narrowed_shape),
        strides_,
        storage_,
        offset_bytes,
        *data_span,
        *narrowed_numel,
        element_size_
    );
}

std::expected<void, TensorError> Tensor::copy_from(const Tensor & source)
{
    if (data_type_ != source.data_type_) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::DataTypeMismatch, "Tensor data type mismatch")
        );
    }

    const auto destination_shape = shape_.values();
    const auto source_shape = source.shape_.values();
    if (destination_shape.size() != source_shape.size() ||
        !std::equal(destination_shape.begin(), destination_shape.end(), source_shape.begin()))
        [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::ShapeMismatch, "Tensor shapes do not match")
        );
    }

    if (numel_ == 0) {
        return {};
    }

    if (!storage_ || !source.storage_) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidByteSize, "Tensor storage is not initialized")
        );
    }

    if (is_contiguous() && source.is_contiguous()) {
        std::memmove(data().data(), source.data().data(), bytes());
        return {};
    }

    if (storage_ == source.storage_) {
        std::vector<std::byte> temporary(bytes());
        for (std::size_t index = 0; index < numel_; ++index) {
            std::memcpy(
                temporary.data() + index * element_size_,
                source.element_pointer(index),
                element_size_
            );
        }
        for (std::size_t index = 0; index < numel_; ++index) {
            std::memcpy(
                element_pointer(index),
                temporary.data() + index * element_size_,
                element_size_
            );
        }
        return {};
    }

    for (std::size_t index = 0; index < numel_; ++index) {
        std::memcpy(element_pointer(index), source.element_pointer(index), element_size_);
    }
    return {};
}

common::data_type::DataType Tensor::data_type() const noexcept
{
    return data_type_;
}

const Shape & Tensor::shape() const noexcept
{
    return shape_;
}

const Strides & Tensor::strides() const noexcept
{
    return strides_;
}

std::size_t Tensor::rank() const noexcept
{
    return shape_.rank();
}

std::size_t Tensor::numel() const noexcept
{
    return numel_;
}

std::size_t Tensor::element_size() const noexcept
{
    return element_size_;
}

std::size_t Tensor::bytes() const noexcept
{
    return numel_ * element_size_;
}

bool Tensor::is_contiguous() const noexcept
{
    const auto dimensions = shape_.values();
    const auto stride_values = strides_.values();
    if (dimensions.size() != stride_values.size()) [[unlikely]] {
        return false;
    }

    std::size_t expected_stride = 1;
    for (std::size_t i = dimensions.size(); i > 0; --i) {
        const std::size_t index = i - 1;
        if (stride_values[index] != expected_stride) [[unlikely]] {
            return false;
        }

        expected_stride *= dimensions[index];
    }

    return true;
}

std::span<std::byte> Tensor::data() noexcept
{
    if (!storage_ || data_offset_bytes_ > storage_->size() ||
        data_span_bytes_ > storage_->size() - data_offset_bytes_) [[unlikely]] {
        return {};
    }

    auto * base = storage_->data();
    if (base == nullptr) {
        return {};
    }
    return std::span<std::byte>(base + data_offset_bytes_, data_span_bytes_);
}

std::span<const std::byte> Tensor::data() const noexcept
{
    if (!storage_ || data_offset_bytes_ > storage_->size() ||
        data_span_bytes_ > storage_->size() - data_offset_bytes_) [[unlikely]] {
        return {};
    }

    const auto * base = storage_->data();
    if (base == nullptr) {
        return {};
    }
    return std::span<const std::byte>(base + data_offset_bytes_, data_span_bytes_);
}

bool Tensor::empty() const noexcept
{
    return numel_ == 0;
}

std::byte * Tensor::element_pointer(std::size_t linear_index) noexcept
{
    const Tensor & self = *this;
    return const_cast<std::byte *>(self.element_pointer(linear_index));
}

const std::byte * Tensor::element_pointer(std::size_t linear_index) const noexcept
{
    if (linear_index >= numel_ || !storage_) [[unlikely]] {
        return nullptr;
    }

    const auto dimensions = shape_.values();
    const auto stride_values = strides_.values();
    std::size_t remaining = linear_index;
    std::size_t element_offset = 0;
    for (std::size_t index = dimensions.size(); index > 0; --index) {
        const std::size_t dimension = index - 1;
        if (dimensions[dimension] == 0) [[unlikely]] {
            return nullptr;
        }

        const std::size_t coordinate = remaining % dimensions[dimension];
        remaining /= dimensions[dimension];
        element_offset += coordinate * stride_values[dimension];
    }

    return storage_->data() + data_offset_bytes_ + element_offset * element_size_;
}

} // namespace liteinfer::core::tensor
