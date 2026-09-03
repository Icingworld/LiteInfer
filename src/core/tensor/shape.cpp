#include "core/tensor/shape.hpp"

#include <limits>
#include <utility>

namespace liteinfer::core::tensor
{

Shape::Shape(std::vector<std::size_t> dimensions)
    : dimensions_(std::move(dimensions))
{}

Shape::Shape(std::initializer_list<std::size_t> dimensions)
    : dimensions_(dimensions)
{}

std::size_t Shape::rank() const noexcept
{
    return dimensions_.size();
}

std::expected<std::size_t, TensorError> Shape::extent(std::size_t index) const
{
    if (index >= dimensions_.size()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::IndexOutOfRange, "Shape index out of range")
        );
    }

    return dimensions_[index];
}

std::expected<std::size_t, TensorError> Shape::numel() const
{
    std::size_t result = 1;

    for (const auto dim : dimensions_) {
        if (dim != 0 && result > std::numeric_limits<std::size_t>::max() / dim) [[unlikely]] {
            return std::unexpected(
                TensorError(TensorErrorCode::Overflow, "Shape dimension overflow")
            );
        }

        result *= dim;
    }

    return result;
}

std::span<const std::size_t> Shape::values() const noexcept
{
    return dimensions_;
}

} // namespace liteinfer::core::tensor
