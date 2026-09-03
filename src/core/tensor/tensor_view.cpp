#include "core/tensor/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace liteinfer::core::tensor
{

std::expected<TensorView, TensorError>
Tensor::narrow(std::size_t dimension, std::size_t start, std::size_t length) &
{
    return as_view().narrow(dimension, start, length);
}

std::expected<ConstTensorView, TensorError>
Tensor::narrow(std::size_t dimension, std::size_t start, std::size_t length) const &
{
    return as_view().narrow(dimension, start, length);
}

namespace detail
{

namespace
{

[[nodiscard]]
bool byte_ranges_overlap(
    std::span<const std::byte> first,
    std::span<const std::byte> second
) noexcept
{
    if (first.empty() || second.empty()) {
        return false;
    }

    const auto first_begin = reinterpret_cast<std::uintptr_t>(first.data());
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data());
    const auto maximum_address = std::numeric_limits<std::uintptr_t>::max();
    if (first.size() > maximum_address - first_begin ||
        second.size() > maximum_address - second_begin) {
        return true;
    }

    const auto first_end = first_begin + first.size();
    const auto second_end = second_begin + second.size();
    return first_begin < second_end && second_begin < first_end;
}

} // namespace

std::expected<void, TensorError> copy_view(
    const BasicTensorView<std::byte> & destination,
    const BasicTensorView<const std::byte> & source
)
{
    if (destination.data_type() != source.data_type()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::DataTypeMismatch, "Tensor data type mismatch")
        );
    }

    const auto destination_shape = destination.shape().values();
    const auto source_shape = source.shape().values();
    if (destination_shape.size() != source_shape.size() ||
        !std::equal(destination_shape.begin(), destination_shape.end(), source_shape.begin()))
        [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::ShapeMismatch, "Tensor shapes do not match")
        );
    }

    if (destination.numel() == 0) {
        return {};
    }

    if (destination.data().empty() || source.data().empty()) [[unlikely]] {
        return std::unexpected(
            TensorError(TensorErrorCode::InvalidByteSize, "Tensor storage is not initialized")
        );
    }

    if (destination.is_contiguous() && source.is_contiguous()) {
        const std::size_t logical_bytes = destination.numel() * destination.element_size();
        std::memmove(destination.data().data(), source.data().data(), logical_bytes);
        return {};
    }

    const std::span<const std::byte> destination_storage = destination.data();
    const std::span<const std::byte> source_storage = source.data();
    if (byte_ranges_overlap(destination_storage, source_storage)) {
        const std::size_t logical_bytes = source.numel() * source.element_size();
        std::vector<std::byte> temporary(logical_bytes);
        for (std::size_t index = 0; index < source.numel(); ++index) {
            std::memcpy(
                temporary.data() + index * source.element_size(),
                source.element_pointer(index),
                source.element_size()
            );
        }
        for (std::size_t index = 0; index < destination.numel(); ++index) {
            std::memcpy(
                destination.element_pointer(index),
                temporary.data() + index * destination.element_size(),
                destination.element_size()
            );
        }
        return {};
    }

    for (std::size_t index = 0; index < destination.numel(); ++index) {
        std::memcpy(
            destination.element_pointer(index),
            source.element_pointer(index),
            destination.element_size()
        );
    }
    return {};
}

} // namespace detail

} // namespace liteinfer::core::tensor
