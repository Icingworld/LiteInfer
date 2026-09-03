#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/data_type/data_type.hpp"
#include "core/tensor/shape.hpp"
#include "core/tensor/strides.hpp"
#include "core/tensor/tensor_error.hpp"

namespace liteinfer::core::tensor
{

// 非持有的 Tensor 视图元素类型概念
// 必须是 std::byte 或 const std::byte，分别表示可写和只读视图
template <typename T>
concept TensorViewElementType =
    std::same_as<T, std::byte> || std::same_as<T, const std::byte>;

template <TensorViewElementType Byte>
class BasicTensorView;

namespace detail
{

[[nodiscard]]
std::expected<void, TensorError> copy_view(
    const BasicTensorView<std::byte> & destination,
    const BasicTensorView<const std::byte> & source
);

[[nodiscard]]
inline std::expected<std::size_t, TensorError>
compute_storage_span_bytes(const Shape & shape, const Strides & strides, std::size_t element_size)
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

} // namespace detail

// 非持有的 Tensor 视图
template <TensorViewElementType Byte>
class BasicTensorView
{
public:
    template <typename T>
    using ViewElementType = std::conditional_t<
        std::is_const_v<Byte> || std::is_const_v<T>,
        const std::remove_cv_t<T>,
        std::remove_cv_t<T>>;

public:
    BasicTensorView(const BasicTensorView & other) = default;

    BasicTensorView & operator=(const BasicTensorView & other) = default;

    BasicTensorView(BasicTensorView && other) noexcept = default;

    BasicTensorView & operator=(BasicTensorView && other) noexcept = default;

    ~BasicTensorView() = default;

    // 可写 view 可以隐式转换为只读 view，但不会复制底层数据
    template <typename OtherByte>
        requires(std::is_const_v<Byte> && std::is_same_v<OtherByte, std::byte>)
    BasicTensorView(const BasicTensorView<OtherByte> & other)
        : data_type_(other.data_type_)
        , shape_(other.shape_)
        , strides_(other.strides_)
        , storage_(other.storage_)
        , numel_(other.numel_)
        , element_size_(other.element_size_)
    {}

public:
    // 获取视图的数据类型
    [[nodiscard]]
    common::data_type::DataType data_type() const noexcept
    {
        return data_type_;
    }

    // 获取视图的形状
    [[nodiscard]]
    const Shape & shape() const noexcept
    {
        return shape_;
    }

    // 获取视图的步长
    [[nodiscard]]
    const Strides & strides() const noexcept
    {
        return strides_;
    }

    // 获取视图的秩
    [[nodiscard]]
    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    // 获取视图的逻辑元素数量
    [[nodiscard]]
    std::size_t numel() const noexcept
    {
        return numel_;
    }

    // 获取视图的元素字节大小
    [[nodiscard]]
    std::size_t element_size() const noexcept
    {
        return element_size_;
    }

    // 视图是否为连续的
    [[nodiscard]]
    bool is_contiguous() const noexcept
    {
        const auto dimensions = shape_.values();
        const auto stride_values = strides_.values();
        if (dimensions.size() != stride_values.size()) [[unlikely]] {
            return false;
        }

        std::size_t expected_stride = 1;
        for (std::size_t index = dimensions.size(); index > 0; --index) {
            const std::size_t dimension = index - 1;
            if (stride_values[dimension] != expected_stride) [[unlikely]] {
                return false;
            }

            if (dimensions[dimension] != 0 &&
                expected_stride > std::numeric_limits<std::size_t>::max() / dimensions[dimension])
                [[unlikely]] {
                return false;
            }
            expected_stride *= dimensions[dimension];
        }

        return true;
    }

    // 获取覆盖视图物理范围的字节视图
    // 非连续视图的物理范围可能包含不属于逻辑视图的间隙，不能按逻辑元素直接解释
    [[nodiscard]]
    std::span<Byte> data() const noexcept
    {
        return storage_;
    }

    // 将视图转换为指定 C++ 元素类型
    // 只有连续视图可以转换为逻辑连续的 std::span<T>
    template <typename T>
    [[nodiscard]]
    std::expected<std::span<ViewElementType<T>>, TensorError> data_as() const
    {
        using ValueType = std::remove_cv_t<T>;

        if constexpr (!requires {
                          common::data_type::DataTypeTraits<ValueType>::value;
                      } || !std::is_trivially_copyable_v<ValueType>) {
            return std::unexpected(
                TensorError(TensorErrorCode::InvalidDataType, "Unsupported C++ tensor data type")
            );
        } else {
            if (data_type_ != common::data_type::DataTypeTraits<ValueType>::value) [[unlikely]] {
                return std::unexpected(
                    TensorError(TensorErrorCode::DataTypeMismatch, "Tensor data type mismatch")
                );
            }

            if (!is_contiguous()) [[unlikely]] {
                return std::unexpected(
                    TensorError(TensorErrorCode::NonContiguous, "Tensor data is not contiguous")
                );
            }

            if (numel_ == 0) {
                return std::span<ViewElementType<T>>();
            }

            const auto raw_data = data();
            const auto address = reinterpret_cast<std::uintptr_t>(raw_data.data());
            if (address % alignof(ValueType) != 0) [[unlikely]] {
                return std::unexpected(TensorError(
                    TensorErrorCode::InvalidDataAlignment,
                    "Tensor data alignment mismatch"
                ));
            }

            return std::span<ViewElementType<T>>(
                reinterpret_cast<ViewElementType<T> *>(raw_data.data()),
                numel_
            );
        }
    }

    // 沿指定维度创建子视图。子视图仍然共享当前视图的底层存储和步长
    [[nodiscard]]
    std::expected<BasicTensorView, TensorError>
    narrow(std::size_t dimension, std::size_t start, std::size_t length) const
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

        std::vector<std::size_t> narrowed_dimensions(dimensions.begin(), dimensions.end());
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

        if (start != 0 && stride_values[dimension] >
                              std::numeric_limits<std::size_t>::max() / start) [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::Overflow, "Tensor narrow offset overflow")
            );
        }
        const std::size_t offset_elements = start * stride_values[dimension];
        if (offset_elements > std::numeric_limits<std::size_t>::max() / element_size_)
            [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::Overflow, "Tensor narrow byte offset overflow")
            );
        }
        const std::size_t offset_bytes = offset_elements * element_size_;

        auto storage_span =
            detail::compute_storage_span_bytes(narrowed_shape, strides_, element_size_);
        if (!storage_span) [[unlikely]] {
            return std::unexpected(std::move(storage_span).error());
        }
        if (offset_bytes > storage_.size() || *storage_span > storage_.size() - offset_bytes)
            [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::IndexOutOfRange, "Tensor narrow range exceeds storage")
            );
        }

        return BasicTensorView(
            data_type_,
            std::move(narrowed_shape),
            strides_,
            storage_.subspan(offset_bytes, *storage_span),
            *narrowed_numel,
            element_size_
        );
    }

    // 将 source 的逻辑元素复制到当前可写 view
    [[nodiscard]]
    std::expected<void, TensorError> copy_from(const BasicTensorView<const std::byte> & source)
        requires(!std::is_const_v<Byte>)
    {
        return detail::copy_view(*this, source);
    }

private:
    BasicTensorView(
        common::data_type::DataType data_type,
        Shape shape,
        Strides strides,
        std::span<Byte> storage,
        std::size_t numel,
        std::size_t element_size
    )
        : data_type_(data_type)
        , shape_(std::move(shape))
        , strides_(std::move(strides))
        , storage_(storage)
        , numel_(numel)
        , element_size_(element_size)
    {}

    [[nodiscard]]
    Byte * element_pointer(std::size_t linear_index) const noexcept
    {
        if (linear_index >= numel_ || numel_ == 0) [[unlikely]] {
            return nullptr;
        }

        const auto dimensions = shape_.values();
        const auto stride_values = strides_.values();
        std::size_t remaining = linear_index;
        std::size_t element_offset = 0;
        for (std::size_t index = dimensions.size(); index > 0; --index) {
            const std::size_t dimension = index - 1;
            const std::size_t coordinate = remaining % dimensions[dimension];
            remaining /= dimensions[dimension];
            element_offset += coordinate * stride_values[dimension];
        }

        return storage_.data() + element_offset * element_size_;
    }

private:
    template <TensorViewElementType>
    friend class BasicTensorView;

    friend class Tensor;

    friend std::expected<void, TensorError> detail::copy_view(
        const BasicTensorView<std::byte> & destination,
        const BasicTensorView<const std::byte> & source
    );

    common::data_type::DataType data_type_;
    Shape shape_;
    Strides strides_;
    std::span<Byte> storage_;
    std::size_t numel_;
    std::size_t element_size_;
};

using TensorView = BasicTensorView<std::byte>;
using ConstTensorView = BasicTensorView<const std::byte>;

} // namespace liteinfer::core::tensor
