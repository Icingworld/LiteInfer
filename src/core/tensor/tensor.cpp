#include "core/tensor/tensor.hpp"

#include <limits>
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
constexpr std::size_t element_size_of(DataType data_type) noexcept
{
    switch (data_type) {
    case DataType::Float16:
        [[fallthrough]];
    case DataType::BFloat16:
        return 2;
    case DataType::Float32:
        return 4;
    case DataType::Int8:
        [[fallthrough]];
    case DataType::Bool:
        return 1;
    case DataType::Int32:
        return 4;
    case DataType::Int64:
        return 8;
    default:
        return 0;
    }
}

// 获取数据类型对应每个元素的字节大小
[[nodiscard]]
std::expected<std::size_t, TensorError> element_size(DataType data_type) noexcept
{
    const std::size_t size = element_size_of(data_type);
    if (size == 0) {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidDataType, "Unsupported tensor data type")
        );
    }

    return size;
}

// 计算 Tensor 的元素个数、元素字节大小与总字节大小
[[nodiscard]]
std::expected<TensorLayout, TensorError>
compute_layout(DataType data_type, const Shape & shape) noexcept
{
    auto numel = shape.numel();
    if (!numel) {
        return std::unexpected(std::move(numel).error());
    }

    auto element_size_result = element_size(data_type);
    if (!element_size_result) {
        return std::unexpected(std::move(element_size_result).error());
    }

    const std::size_t count = *numel;
    const std::size_t element_bytes = *element_size_result;
    if (count != 0 && element_bytes > std::numeric_limits<std::size_t>::max() / count) {
        return std::unexpected(TensorError(TensorErrorCode::Overflow, "Tensor byte size overflow"));
    }

    return TensorLayout {
        .numel = count,
        .element_size = element_bytes,
        .bytes = count * element_bytes,
    };
}

} // namespace

Tensor::Tensor(
    DataType data_type,
    Shape shape,
    Strides strides,
    std::vector<std::byte> data,
    std::size_t numel,
    std::size_t element_size
)
    : data_type_(data_type)
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , data_(std::move(data))
    , numel_(numel)
    , element_size_(element_size)
{}

std::expected<Tensor, TensorError> Tensor::allocate(DataType data_type, Shape shape)
{
    auto layout = compute_layout(data_type, shape);
    if (!layout) {
        return std::unexpected(std::move(layout).error());
    }

    std::vector<std::byte> data(layout->bytes, std::byte {0});

    auto strides = Strides::from_shape(shape);
    if (!strides) {
        return std::unexpected(std::move(strides).error());
    }

    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(data),
        layout->numel,
        layout->element_size
    );
}

std::expected<Tensor, TensorError>
Tensor::from_bytes(DataType data_type, Shape shape, std::span<const std::byte> bytes)
{
    auto layout = compute_layout(data_type, shape);
    if (!layout) {
        return std::unexpected(std::move(layout).error());
    }

    if (bytes.size() != layout->bytes) {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidByteSize, "Tensor byte size mismatch")
        );
    }

    std::vector<std::byte> data(bytes.begin(), bytes.end());

    auto strides = Strides::from_shape(shape);
    if (!strides) {
        return std::unexpected(std::move(strides).error());
    }

    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(data),
        layout->numel,
        layout->element_size
    );
}

DataType Tensor::data_type() const noexcept
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
    return data_.size();
}

bool Tensor::is_contiguous() const noexcept
{
    const auto dimensions = shape_.values();
    const auto stride_values = strides_.values();
    if (dimensions.size() != stride_values.size()) {
        return false;
    }

    std::size_t expected_stride = 1;
    for (std::size_t i = dimensions.size(); i > 0; --i) {
        const std::size_t index = i - 1;
        if (stride_values[index] != expected_stride) {
            return false;
        }

        expected_stride *= dimensions[index];
    }

    return true;
}

std::span<std::byte> Tensor::data() noexcept
{
    return data_;
}

std::span<const std::byte> Tensor::data() const noexcept
{
    return data_;
}

bool Tensor::empty() const noexcept
{
    return numel_ == 0;
}

} // namespace liteinfer::core::tensor
