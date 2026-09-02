#include "core/tensor/strides.hpp"

#include <limits>
#include <utility>

namespace liteinfer::core::tensor
{

Strides::Strides(std::vector<std::size_t> strides)
    : strides_(std::move(strides))
{}

Strides::Strides(std::initializer_list<std::size_t> strides)
    : strides_(strides)
{}

std::expected<Strides, TensorError> Strides::row_major(const Shape & shape)
{
    const auto dimensions = shape.values();
    std::vector<std::size_t> strides(dimensions.size());

    std::size_t stride = 1;
    for (std::size_t i = dimensions.size(); i > 0; --i) {
        const std::size_t index = i - 1;
        strides[index] = stride;

        const std::size_t dim = dimensions[index];
        if (dim != 0 && stride > std::numeric_limits<std::size_t>::max() / dim) [[unlikely]] {
            return std::unexpected(TensorError(TensorErrorCode::Overflow, "Strides overflow"));
        }
        stride *= dim;
    }

    return Strides(std::move(strides));
}

std::size_t Strides::rank() const noexcept
{
    return strides_.size();
}

std::expected<std::size_t, TensorError> Strides::stride(std::size_t index) const noexcept
{
    if (index >= strides_.size()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::IndexOutOfRange, "Strides index out of range")
        );
    }

    return strides_[index];
}

std::span<const std::size_t> Strides::values() const noexcept
{
    return strides_;
}

} // namespace liteinfer::core::tensor
