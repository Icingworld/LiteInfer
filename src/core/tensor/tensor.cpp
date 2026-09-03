#include "core/tensor/tensor.hpp"

#include <limits>
#include <utility>

namespace liteinfer::core::tensor
{

namespace
{

// 张量布局
struct TensorLayout
{
    std::size_t numel;
    std::size_t element_size;
    std::size_t bytes;
};

// 计算 Tensor 的元素个数、元素字节大小与总字节大小
[[nodiscard]]
std::expected<TensorLayout, TensorError>
compute_layout(common::data_type::DataType data_type, const Shape & shape)
{
    auto numel = shape.numel();
    if (!numel) [[unlikely]] {
        return std::unexpected(std::move(numel).error());
    }

    std::size_t element_size = 0;
    switch (data_type) {
    case common::data_type::DataType::Float16:
        [[fallthrough]];
    case common::data_type::DataType::BFloat16:
        element_size = 2;
        break;
    case common::data_type::DataType::Float32:
        element_size = 4;
        break;
    case common::data_type::DataType::Int8:
        [[fallthrough]];
    case common::data_type::DataType::Bool:
        element_size = 1;
        break;
    case common::data_type::DataType::Int32:
        element_size = 4;
        break;
    case common::data_type::DataType::Int64:
        element_size = 8;
        break;
    [[unlikely]]
    default:
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidDataType, "Unsupported tensor data type")
        );
    }

    const std::size_t count = *numel;
    if (count != 0 && element_size > std::numeric_limits<std::size_t>::max() / count) [[unlikely]] {
        return std::unexpected(TensorError(TensorErrorCode::Overflow, "Tensor byte size overflow"));
    }

    return TensorLayout {
        .numel = count,
        .element_size = element_size,
        .bytes = count * element_size,
    };
}

} // namespace

Tensor::Tensor(
    common::data_type::DataType data_type,
    Shape shape,
    Strides strides,
    std::vector<std::byte> storage,
    std::size_t numel,
    std::size_t element_size
)
    : data_type_(data_type)
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , storage_(std::move(storage))
    , numel_(numel)
    , element_size_(element_size)
{}

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

    auto storage = std::vector<std::byte>(layout->bytes, std::byte {0});

    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(storage),
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

    auto storage = std::vector<std::byte>(bytes.begin(), bytes.end());

    return Tensor(
        data_type,
        std::move(shape),
        *std::move(strides),
        std::move(storage),
        layout->numel,
        layout->element_size
    );
}

TensorView Tensor::as_view() &
{
    return TensorView(data_type_, shape_, strides_, data(), numel_, element_size_);
}

ConstTensorView Tensor::as_view() const &
{
    return ConstTensorView(data_type_, shape_, strides_, data(), numel_, element_size_);
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

        // 这里不检查乘法是否会溢出，因为在构造阶段保证了 Tensor 合法
        expected_stride *= dimensions[index];
    }

    return true;
}

std::span<std::byte> Tensor::data() noexcept
{
    return std::span<std::byte>(storage_.data(), storage_.size());
}

std::span<const std::byte> Tensor::data() const noexcept
{
    return std::span<const std::byte>(storage_.data(), storage_.size());
}

std::expected<void, TensorError> Tensor::copy_from(ConstTensorView source)
{
    return detail::copy_view(as_view(), source);
}

} // namespace liteinfer::core::tensor
