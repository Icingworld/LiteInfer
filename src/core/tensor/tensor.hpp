#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/tensor/data_type.hpp"
#include "core/tensor/shape.hpp"
#include "core/tensor/strides.hpp"
#include "core/tensor/tensor_error.hpp"

namespace liteinfer::core::tensor
{

// Tensor
// 当前 Tensor 是默认以 row-major 方式连续存储的
// 默认深拷贝、可移动
class Tensor
{
private:
    explicit Tensor(
        DataType data_type,
        Shape shape,
        Strides strides,
        std::vector<std::byte> data,
        std::size_t numel,
        std::size_t element_size
    );

public:
    // 分配内存并初始化 Tensor
    [[nodiscard]]
    static std::expected<Tensor, TensorError> allocate(DataType data_type, Shape shape);

    // 从字节数组创建 Tensor
    [[nodiscard]]
    static std::expected<Tensor, TensorError>
    from_bytes(DataType data_type, Shape shape, std::span<const std::byte> bytes);

    // 获取 Tensor 的数据类型
    [[nodiscard]]
    DataType data_type() const noexcept;

    // 获取 Tensor 的 Shape
    [[nodiscard]]
    const Shape & shape() const noexcept;

    // 获取 Tensor 的 Strides
    [[nodiscard]]
    const Strides & strides() const noexcept;

    // 获取 Tensor 的秩
    [[nodiscard]]
    std::size_t rank() const noexcept;

    // 获取 Tensor 中一共有多少元素
    [[nodiscard]]
    std::size_t numel() const noexcept;

    // 获取 Tensor 中每个元素的字节大小
    [[nodiscard]]
    std::size_t element_size() const noexcept;

    // 获取 Tensor 中所有元素的字节大小
    [[nodiscard]]
    std::size_t bytes() const noexcept;

    // Tensor 的元素数量是否为空
    [[nodiscard]]
    bool empty() const noexcept;

    // Tensor 是否为连续的
    [[nodiscard]]
    bool is_contiguous() const noexcept;

    // 获取 Tensor 的数据视图
    [[nodiscard]]
    std::span<std::byte> data() noexcept;

    // 获取 Tensor 的只读数据视图
    [[nodiscard]]
    std::span<const std::byte> data() const noexcept;

    // 获取 Tensor 的数据视图，并转换为指定类型，如果类型不一致则返回错误
    template <typename T>
    [[nodiscard]]
    std::expected<std::span<T>, TensorError> data_as();

    // 获取 Tensor 的只读数据视图，并转换为指定类型，如果类型不一致则返回错误
    template <typename T>
    [[nodiscard]]
    std::expected<std::span<const T>, TensorError> data_as() const;

private:
    DataType data_type_;
    Shape shape_;
    Strides strides_;
    std::vector<std::byte> data_;
    std::size_t numel_;
    std::size_t element_size_;
};

template <typename T>
std::expected<std::span<const T>, TensorError> Tensor::data_as() const
{
    using ValueType = std::remove_cv_t<T>;

    if constexpr (!requires { DataTypeOf<ValueType>::value; }) {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidDataType, "Unsupported C++ tensor data type")
        );
    } else if constexpr (!std::is_trivially_copyable_v<ValueType>) {
        return std::unexpected(TensorError(
            TensorErrorCode::InvalidDataType,
            "Tensor data type is not trivially copyable"
        ));
    } else {
        if (data_type_ != DataTypeOf<ValueType>::value) {
            return std::unexpected(
                TensorError(TensorErrorCode::DataTypeMismatch, "Tensor data type mismatch")
            );
        }

        if (!is_contiguous()) {
            return std::unexpected(
                TensorError(TensorErrorCode::NonContiguous, "Tensor data is not contiguous")
            );
        }

        if (numel_ == 0) {
            return std::span<const T>();
        }

        const auto address = reinterpret_cast<std::uintptr_t>(data_.data());
        if (address % alignof(ValueType) != 0) {
            return std::unexpected(
                TensorError(TensorErrorCode::InvalidDataAlignment, "Tensor data alignment mismatch")
            );
        }

        return std::span<const T>(reinterpret_cast<const T *>(data_.data()), numel_);
    }
}

template <typename T>
std::expected<std::span<T>, TensorError> Tensor::data_as()
{
    using ValueType = std::remove_cv_t<T>;

    const Tensor & self = *this;
    auto view = self.data_as<ValueType>();
    if (!view) {
        return std::unexpected(std::move(view).error());
    }

    return std::span<T>(const_cast<ValueType *>(view->data()), view->size());
}

} // namespace liteinfer::core::tensor
