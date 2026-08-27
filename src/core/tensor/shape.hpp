#pragma once

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <span>
#include <vector>

#include "core/tensor/tensor_error.hpp"

namespace liteinfer::core::tensor
{

// Tensor 形状
class Shape
{
public:
    explicit Shape(std::vector<std::size_t> dimensions);

    explicit Shape(std::initializer_list<std::size_t> dimensions);

public:
    // 获取 Shape 的秩
    [[nodiscard]]
    std::size_t rank() const noexcept;

    // 获取 Shape 指定维度的长度，如果索引越界则返回错误
    [[nodiscard]]
    std::expected<std::size_t, TensorError> extent(std::size_t index) const noexcept;

    // 计算 Tensor 中一共有多少元素，如果计算失败则返回错误
    [[nodiscard]]
    std::expected<std::size_t, TensorError> numel() const noexcept;

    // 获取 Shape 的视图
    [[nodiscard]]
    std::span<const std::size_t> values() const noexcept;

    // Shape 是否为空
    [[nodiscard]]
    bool empty() const noexcept;

private:
    std::vector<std::size_t> dimensions_;
};

} // namespace liteinfer::core::tensor
