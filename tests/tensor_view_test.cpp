#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core::tensor;

Tensor make_float_tensor(Shape shape, std::size_t value_count)
{
    auto result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Float32, std::move(shape));
    assert(result.has_value());

    auto values = result->data_as<float>();
    assert(values.has_value());
    assert(values->size() == value_count);
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = static_cast<float>(index);
    }
    return std::move(*result);
}

void test_narrow_layout_and_const_propagation()
{
    auto tensor = make_float_tensor(Shape {2, 3, 4}, 24);

    auto view = tensor.narrow(1, 1, 2);
    assert(view.has_value());
    assert(view->data_type() == liteinfer::core::common::data_type::DataType::Float32);
    assert(view->rank() == 3);
    assert(view->numel() == 16);
    assert(view->numel() * view->element_size() == 16 * sizeof(float));
    assert(view->data().size() == 20 * sizeof(float));
    assert(!view->is_contiguous());

    const auto shape = view->shape().values();
    assert(shape.size() == 3);
    assert(shape[0] == 2);
    assert(shape[1] == 2);
    assert(shape[2] == 4);

    const auto strides = view->strides().values();
    assert(strides.size() == 3);
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);

    auto non_contiguous_values = view->data_as<float>();
    assert(!non_contiguous_values.has_value());
    assert(
        non_contiguous_values.error().code() == std::to_underlying(TensorErrorCode::NonContiguous)
    );

    const Tensor & const_tensor = tensor;
    auto const_view = const_tensor.narrow(1, 1, 2);
    assert(const_view.has_value());
    static_assert(std::is_same_v<decltype(const_view->data()), std::span<const std::byte>>);

    auto const_values = const_view->data_as<float>();
    assert(!const_values.has_value());
    assert(const_values.error().code() == std::to_underlying(TensorErrorCode::NonContiguous));
}

void test_contiguous_subview_access()
{
    auto tensor = make_float_tensor(Shape {2, 3, 4}, 24);

    auto view = tensor.narrow(0, 1, 1);
    assert(view.has_value());
    assert(view->is_contiguous());
    assert(view->data().size() == view->numel() * view->element_size());

    auto values = view->data_as<float>();
    assert(values.has_value());
    assert(values->size() == 12);
    assert((*values)[0] == 12.0F);
    assert((*values)[11] == 23.0F);
}

void test_copy_between_tensor_and_view()
{
    auto destination = make_float_tensor(Shape {1, 2, 4, 2}, 16);
    auto source = make_float_tensor(Shape {1, 2, 2, 2}, 8);

    auto destination_view = destination.narrow(2, 1, 2);
    assert(destination_view.has_value());
    assert(!destination_view->is_contiguous());
    assert(destination_view->copy_from(source.as_view()).has_value());

    auto destination_values = destination.data_as<float>();
    assert(destination_values.has_value());
    assert((*destination_values)[2] == 0.0F);
    assert((*destination_values)[3] == 1.0F);
    assert((*destination_values)[4] == 2.0F);
    assert((*destination_values)[5] == 3.0F);
    assert((*destination_values)[10] == 4.0F);
    assert((*destination_values)[11] == 5.0F);
    assert((*destination_values)[12] == 6.0F);
    assert((*destination_values)[13] == 7.0F);

    auto materialized = Tensor::allocate(destination_view->data_type(), destination_view->shape());
    assert(materialized.has_value());
    assert(materialized->copy_from(*destination_view).has_value());

    auto materialized_values = materialized->data_as<float>();
    assert(materialized_values.has_value());
    for (std::size_t index = 0; index < materialized_values->size(); ++index) {
        assert((*materialized_values)[index] == static_cast<float>(index));
    }
}

void test_narrow_validation()
{
    auto tensor = make_float_tensor(Shape {2, 3}, 6);

    auto invalid_dimension = tensor.narrow(2, 0, 1);
    assert(!invalid_dimension.has_value());
    assert(
        invalid_dimension.error().code() == std::to_underlying(TensorErrorCode::IndexOutOfRange)
    );

    auto invalid_start = tensor.narrow(1, 4, 0);
    assert(!invalid_start.has_value());
    assert(invalid_start.error().code() == std::to_underlying(TensorErrorCode::IndexOutOfRange));

    auto invalid_length = tensor.narrow(1, 2, 2);
    assert(!invalid_length.has_value());
    assert(invalid_length.error().code() == std::to_underlying(TensorErrorCode::IndexOutOfRange));
}

void test_overlapping_copy()
{
    auto contiguous = make_float_tensor(Shape {8}, 8);
    auto source = contiguous.narrow(0, 0, 6);
    auto destination = contiguous.narrow(0, 1, 6);
    assert(source.has_value());
    assert(destination.has_value());
    assert(destination->copy_from(*source).has_value());

    auto contiguous_values = contiguous.data_as<float>();
    assert(contiguous_values.has_value());
    const float expected_contiguous[] {0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 7.0F};
    for (std::size_t index = 0; index < 8; ++index) {
        assert((*contiguous_values)[index] == expected_contiguous[index]);
    }

    auto strided = make_float_tensor(Shape {3, 4}, 12);
    auto strided_source = strided.narrow(1, 0, 3);
    auto strided_destination = strided.narrow(1, 1, 3);
    assert(strided_source.has_value());
    assert(strided_destination.has_value());
    assert(!strided_source->is_contiguous());
    assert(!strided_destination->is_contiguous());
    assert(strided_destination->copy_from(*strided_source).has_value());

    auto strided_values = strided.data_as<float>();
    assert(strided_values.has_value());
    const float expected_strided[] {
        0.0F,
        0.0F,
        1.0F,
        2.0F,
        4.0F,
        4.0F,
        5.0F,
        6.0F,
        8.0F,
        8.0F,
        9.0F,
        10.0F,
    };
    for (std::size_t index = 0; index < 12; ++index) {
        assert((*strided_values)[index] == expected_strided[index]);
    }
}

void test_copy_validation()
{
    auto destination = make_float_tensor(Shape {2, 2}, 4);
    auto wrong_shape = make_float_tensor(Shape {4}, 4);
    auto wrong_dtype_result =
        Tensor::allocate(liteinfer::core::common::data_type::DataType::Int32, Shape {2, 2});
    assert(wrong_dtype_result.has_value());

    assert(!destination.copy_from(wrong_shape.as_view()).has_value());
    assert(!destination.copy_from(wrong_dtype_result->as_view()).has_value());
}

} // namespace

int main()
{
    test_narrow_layout_and_const_propagation();
    test_contiguous_subview_access();
    test_copy_between_tensor_and_view();
    test_narrow_validation();
    test_overlapping_copy();
    test_copy_validation();
}
