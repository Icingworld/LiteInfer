#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/data_type/data_type.hpp"
#include "core/tensor/shape.hpp"
#include "core/tensor/strides.hpp"

namespace liteinfer::core::tensor
{

// 张量元素类型概念
// 必须能够被 common::data_type::DataTypeTraits 识别
// 且是平凡可复制类型，当前实现中的类型均满足要求
template <typename T>
concept TensorElementType = requires {
    common::data_type::DataTypeTraits<std::remove_cv_t<T>>::value;
} && std::is_trivially_copyable_v<std::remove_cv_t<T>>;

// 张量
class Tensor
{
public:
    Tensor(const Tensor & other) = default;

    Tensor & operator=(const Tensor & other) = default;

    Tensor(Tensor && other) noexcept = default;

    Tensor & operator=(Tensor && other) noexcept = default;

    ~Tensor() = default;

private:
    explicit Tensor(
        common::data_type::DataType data_type,
        Shape shape,
        Strides strides,
        std::vector<std::byte> storage,
        std::size_t numel,
        std::size_t element_size
    );

public:
    // 分配内存并零初始化张量
    [[nodiscard]]
    static std::expected<Tensor, TensorError> allocate(common::data_type::DataType data_type, Shape shape);

    // 从字节数组创建张量
    [[nodiscard]]
    static std::expected<Tensor, TensorError> from_bytes(
        common::data_type::DataType data_type,
        Shape shape,
        std::span<const std::byte> bytes
    );

    // 获取张量的数据类型
    [[nodiscard]]
    common::data_type::DataType data_type() const noexcept;

    // 获取张量的形状
    [[nodiscard]]
    const Shape & shape() const noexcept;

    // 获取张量的步长
    [[nodiscard]]
    const Strides & strides() const noexcept;

    // 获取张量的秩
    [[nodiscard]]
    std::size_t rank() const noexcept;

    // 获取张量的总元素数量
    [[nodiscard]]
    std::size_t numel() const noexcept;

    // 获取张量的元素类型大小
    [[nodiscard]]
    std::size_t element_size() const noexcept;

    // 张量是否为连续
    [[nodiscard]]
    bool is_contiguous() const noexcept;

    // 获取张量的底层数据视图
    [[nodiscard]]
    std::span<std::byte> data() noexcept;

    // 获取张量的底层数据只读视图
    [[nodiscard]]
    std::span<const std::byte> data() const noexcept;

    // 获取张量的底层数据视图，并转换为指定类型，如果类型不一致则返回错误
    template <TensorElementType T>
    [[nodiscard]]
    std::expected<std::span<T>, TensorError> data_as();

    // 获取张量的底层数据只读视图，并转换为指定类型，如果类型不一致则返回错误
    template <TensorElementType T>
    [[nodiscard]]
    std::expected<std::span<const T>, TensorError> data_as() const;

private:
    common::data_type::DataType data_type_;
    Shape shape_;
    Strides strides_;
    std::vector<std::byte> storage_;
    std::size_t numel_;
    std::size_t element_size_;
};

template <TensorElementType T>
std::expected<std::span<const T>, TensorError> Tensor::data_as() const
{
    using ValueType = std::remove_cv_t<T>;

    if (data_type_ != common::data_type::DataTypeTraits<ValueType>::value) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::DataTypeMismatch, "Tensor data type mismatch")
        );
    }

    if (numel_ == 0) {
        return std::span<const T>();
    }

    const auto raw_data = data();
    const auto address = reinterpret_cast<std::uintptr_t>(raw_data.data());
    if (address % alignof(ValueType) != 0) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidDataAlignment, "Tensor data alignment mismatch")
        );
    }

    return std::span<const T>(reinterpret_cast<const T *>(raw_data.data()), numel_);
}

template <TensorElementType T>
std::expected<std::span<T>, TensorError> Tensor::data_as()
{
    using ValueType = std::remove_cv_t<T>;

    const Tensor & self = *this;
    auto view = self.data_as<ValueType>();
    if (!view) [[unlikely]] {
        return std::unexpected(std::move(view).error());
    }

    return std::span<T>(const_cast<ValueType *>(view->data()), view->size());
}

} // namespace liteinfer::core::tensor
