#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <utility>

#include "core/layer/layer_error.hpp"
#include "core/layer/rms_norm.hpp"

namespace
{

using namespace liteinfer::core::layer;
using namespace liteinfer::core::tensor;

Tensor make_float_tensor(Shape shape, std::initializer_list<float> values)
{
    auto result = Tensor::allocate(DataType::Float32, std::move(shape));
    assert(result.has_value());

    auto data = result->data_as<float>();
    assert(data.has_value());
    assert(data->size() == values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*data)[i] = values.begin()[i];
    }

    return std::move(*result);
}

void assert_near(float actual, float expected, float tolerance = 1.0e-5F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void assert_float_values_near(const Tensor & tensor, std::initializer_list<float> expected)
{
    auto values = tensor.data_as<float>();
    assert(values.has_value());
    assert(values->size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert_near((*values)[i], expected.begin()[i]);
    }
}

void test_forward_preserves_shape_and_normalizes_each_row()
{
    auto weight = make_float_tensor(Shape {2}, {1.0F, 2.0F});
    auto input = make_float_tensor(Shape {2, 1, 2}, {3.0F, 4.0F, -1.0F, 1.0F});

    auto rms_norm = RMSNorm::create(std::move(weight), 1.0F);
    assert(rms_norm.has_value());
    assert(rms_norm->normalized_size() == 2);
    assert(rms_norm->eps() == 1.0F);

    auto output = rms_norm->forward(input);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 2);
    assert(*output->shape().extent(1) == 1);
    assert(*output->shape().extent(2) == 2);

    const float first_inverse_rms = 1.0F / std::sqrt(13.5F);
    const float second_inverse_rms = 1.0F / std::sqrt(2.0F);
    assert_float_values_near(
        *output,
        {
            3.0F * first_inverse_rms,
            8.0F * first_inverse_rms,
            -second_inverse_rms,
            2.0F * second_inverse_rms,
        }
    );
}

void test_empty_input()
{
    auto weight = make_float_tensor(Shape {3}, {1.0F, 1.0F, 1.0F});
    auto input_result = Tensor::allocate(DataType::Float32, Shape {0, 2, 3});
    assert(input_result.has_value());

    auto rms_norm = RMSNorm::create(std::move(weight), 1.0e-6F);
    assert(rms_norm.has_value());

    auto output = rms_norm->forward(*input_result);
    assert(output.has_value());
    assert(output->empty());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 0);
    assert(*output->shape().extent(1) == 2);
    assert(*output->shape().extent(2) == 3);
}

void test_create_validation()
{
    auto rank_two_weight = make_float_tensor(Shape {1, 2}, {1.0F, 2.0F});
    auto rank_two_result = RMSNorm::create(std::move(rank_two_weight), 1.0e-6F);
    assert(!rank_two_result.has_value());
    assert(rank_two_result.error().code() == std::to_underlying(LayerErrorCode::InvalidWeight));

    auto empty_weight = make_float_tensor(Shape {0}, {});
    auto empty_result = RMSNorm::create(std::move(empty_weight), 1.0e-6F);
    assert(!empty_result.has_value());
    assert(empty_result.error().code() == std::to_underlying(LayerErrorCode::InvalidWeight));

    auto integer_weight_result = Tensor::allocate(DataType::Int32, Shape {2});
    assert(integer_weight_result.has_value());
    auto integer_result = RMSNorm::create(std::move(*integer_weight_result), 1.0e-6F);
    assert(!integer_result.has_value());
    assert(
        integer_result.error().code() == std::to_underlying(LayerErrorCode::UnsupportedDataType)
    );

    auto zero_eps_weight = make_float_tensor(Shape {2}, {1.0F, 1.0F});
    auto zero_eps_result = RMSNorm::create(std::move(zero_eps_weight), 0.0F);
    assert(!zero_eps_result.has_value());
    assert(zero_eps_result.error().code() == std::to_underlying(LayerErrorCode::InvalidEpsilon));

    auto nan_eps_weight = make_float_tensor(Shape {2}, {1.0F, 1.0F});
    auto nan_eps_result =
        RMSNorm::create(std::move(nan_eps_weight), std::numeric_limits<float>::quiet_NaN());
    assert(!nan_eps_result.has_value());
    assert(nan_eps_result.error().code() == std::to_underlying(LayerErrorCode::InvalidEpsilon));

    auto infinite_eps_weight = make_float_tensor(Shape {2}, {1.0F, 1.0F});
    auto infinite_eps_result =
        RMSNorm::create(std::move(infinite_eps_weight), std::numeric_limits<float>::infinity());
    assert(!infinite_eps_result.has_value());
    assert(
        infinite_eps_result.error().code() == std::to_underlying(LayerErrorCode::InvalidEpsilon)
    );
}

void test_forward_validation()
{
    auto weight = make_float_tensor(Shape {2}, {1.0F, 1.0F});
    auto rms_norm = RMSNorm::create(std::move(weight), 1.0e-6F);
    assert(rms_norm.has_value());

    auto wrong_shape_input = make_float_tensor(Shape {1, 3}, {1.0F, 2.0F, 3.0F});
    auto wrong_shape_result = rms_norm->forward(wrong_shape_input);
    assert(!wrong_shape_result.has_value());
    assert(wrong_shape_result.error().code() == std::to_underlying(LayerErrorCode::InvalidInput));

    auto integer_input_result = Tensor::allocate(DataType::Int32, Shape {1, 2});
    assert(integer_input_result.has_value());
    auto integer_result = rms_norm->forward(*integer_input_result);
    assert(!integer_result.has_value());
    assert(
        integer_result.error().code() == std::to_underlying(LayerErrorCode::UnsupportedDataType)
    );

    auto scalar_input = make_float_tensor(Shape {}, {1.0F});
    auto scalar_result = rms_norm->forward(scalar_input);
    assert(!scalar_result.has_value());
    assert(scalar_result.error().code() == std::to_underlying(LayerErrorCode::InvalidInput));
}

} // namespace

int main()
{
    test_forward_preserves_shape_and_normalizes_each_row();
    test_empty_input();
    test_create_validation();
    test_forward_validation();
}
