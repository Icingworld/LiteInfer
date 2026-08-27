#include <cassert>
#include <cstdint>
#include <utility>

#include "core/embedding/embedding.hpp"

namespace
{

using namespace liteinfer::core::embedding;
using namespace liteinfer::core::tensor;

Embedding make_embedding()
{
    auto weight_result = Tensor::allocate(DataType::Float32, Shape {4, 3});
    assert(weight_result.has_value());

    auto weight_values = weight_result->data_as<float>();
    assert(weight_values.has_value());
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            (*weight_values)[row * 3 + column] = static_cast<float>(row * 10 + column);
        }
    }

    auto embedding_result = Embedding::create(std::move(*weight_result));
    assert(embedding_result.has_value());
    return std::move(*embedding_result);
}

void test_one_dimensional_input()
{
    auto embedding = make_embedding();
    auto input_result = Tensor::allocate(DataType::Int32, Shape {3});
    assert(input_result.has_value());

    auto input_values = input_result->data_as<std::int32_t>();
    assert(input_values.has_value());
    (*input_values)[0] = 2;
    (*input_values)[1] = 0;
    (*input_values)[2] = 3;

    auto output = embedding.forward(*input_result);
    assert(output.has_value());
    assert(output->shape().rank() == 2);
    assert(*output->shape().extent(0) == 3);
    assert(*output->shape().extent(1) == 3);

    auto output_values = output->data_as<float>();
    assert(output_values.has_value());
    const float expected[] = {20.0F, 21.0F, 22.0F, 0.0F, 1.0F, 2.0F, 30.0F, 31.0F, 32.0F};
    for (std::size_t i = 0; i < 9; ++i) {
        assert((*output_values)[i] == expected[i]);
    }
}

void test_two_dimensional_input()
{
    auto embedding = make_embedding();
    auto input_result = Tensor::allocate(DataType::Int64, Shape {2, 2});
    assert(input_result.has_value());

    auto input_values = input_result->data_as<std::int64_t>();
    assert(input_values.has_value());
    (*input_values)[0] = 0;
    (*input_values)[1] = 1;
    (*input_values)[2] = 2;
    (*input_values)[3] = 3;

    auto output = embedding.forward(*input_result);
    assert(output.has_value());
    assert(output->shape().rank() == 3);
    assert(*output->shape().extent(0) == 2);
    assert(*output->shape().extent(1) == 2);
    assert(*output->shape().extent(2) == 3);

    auto output_values = output->data_as<float>();
    assert(output_values.has_value());
    const float expected[] = {
        0.0F,
        1.0F,
        2.0F,
        10.0F,
        11.0F,
        12.0F,
        20.0F,
        21.0F,
        22.0F,
        30.0F,
        31.0F,
        32.0F,
    };
    for (std::size_t i = 0; i < 12; ++i) {
        assert((*output_values)[i] == expected[i]);
    }
}

void test_invalid_input()
{
    auto embedding = make_embedding();

    auto negative_result = Tensor::allocate(DataType::Int32, Shape {1});
    assert(negative_result.has_value());
    auto negative_values = negative_result->data_as<std::int32_t>();
    assert(negative_values.has_value());
    (*negative_values)[0] = -1;

    auto negative_output = embedding.forward(*negative_result);
    assert(!negative_output.has_value());
    assert(
        negative_output.error().code() == std::to_underlying(EmbeddingErrorCode::InvalidTokenId)
    );

    auto out_of_range_result = Tensor::allocate(DataType::Int64, Shape {1});
    assert(out_of_range_result.has_value());
    auto out_of_range_values = out_of_range_result->data_as<std::int64_t>();
    assert(out_of_range_values.has_value());
    (*out_of_range_values)[0] = 4;

    auto out_of_range_output = embedding.forward(*out_of_range_result);
    assert(!out_of_range_output.has_value());
    assert(
        out_of_range_output.error().code() == std::to_underlying(EmbeddingErrorCode::InvalidTokenId)
    );

    auto wrong_type_result = Tensor::allocate(DataType::Float32, Shape {1});
    assert(wrong_type_result.has_value());
    auto wrong_type_output = embedding.forward(*wrong_type_result);
    assert(!wrong_type_output.has_value());
    assert(
        wrong_type_output.error().code() ==
        std::to_underlying(EmbeddingErrorCode::UnsupportedInputDataType)
    );

    auto wrong_rank_result = Tensor::allocate(DataType::Int32, Shape {1, 1, 1});
    assert(wrong_rank_result.has_value());
    auto wrong_rank_output = embedding.forward(*wrong_rank_result);
    assert(!wrong_rank_output.has_value());
    assert(
        wrong_rank_output.error().code() == std::to_underlying(EmbeddingErrorCode::InvalidInputRank)
    );
}

} // namespace

int main()
{
    test_one_dimensional_input();
    test_two_dimensional_input();
    test_invalid_input();
}
