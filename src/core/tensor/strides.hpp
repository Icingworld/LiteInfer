#pragma once

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <span>
#include <vector>

#include "core/tensor/shape.hpp"
#include "core/tensor/tensor_error.hpp"

namespace liteinfer::core::tensor
{

// Tensor 步长
class Strides
{
public:
    explicit Strides(std::vector<std::size_t> strides);

    explicit Strides(std::initializer_list<std::size_t> strides);

public:
    // 从 Shape 创建 row-major 步长，如果创建失败则返回错误
    // 用于以 row-major 来描述一段连续的内存
    [[nodiscard]]
    static std::expected<Strides, TensorError> row_major(const Shape & shape);

    // 获取 Strides 的秩
    [[nodiscard]]
    std::size_t rank() const noexcept;

    // 获取 Strides 指定维度的步长，如果索引越界则返回错误
    [[nodiscard]]
    std::expected<std::size_t, TensorError> stride(std::size_t index) const noexcept;

    // 获取 Strides 的底层数据视图
    [[nodiscard]]
    std::span<const std::size_t> values() const noexcept;

private:
    std::vector<std::size_t> strides_;
};

} // namespace liteinfer::core::tensor
