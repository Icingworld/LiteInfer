#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/kernels/matadd.hpp"
#include "core/kernels/matmul.hpp"
#include "core/kernels/rms_norm.hpp"
#include "core/kernels/rope.hpp"
#include "core/kernels/sdpa.hpp"
#include "core/kernels/silu_mul.hpp"

namespace
{

using namespace liteinfer::core::kernels;

void assert_near(float actual, float expected, float tolerance = 1.0e-5F)
{
    assert(std::fabs(actual - expected) <= tolerance);
}

void assert_vector_near(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float tolerance = 1.0e-5F
)
{
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        assert_near(actual[i], expected[i], tolerance);
    }
}

void test_matadd()
{
    const std::vector<float> lhs {1.0F, -2.0F, 3.0F, 4.0F};
    const std::vector<float> rhs {5.0F, 6.0F, -7.0F, 8.0F};
    std::vector<float> output(lhs.size());

    matadd_f32(lhs, rhs, output);

    assert_vector_near(output, {6.0F, 4.0F, -4.0F, 12.0F});
}

void test_matmul_rectangular_matrices()
{
    const std::vector<float> lhs {
        1.0F,
        2.0F,
        3.0F,
        4.0F,
        5.0F,
        6.0F,
    };
    const std::vector<float> rhs {
        7.0F,
        8.0F,
        9.0F,
        10.0F,
        11.0F,
        12.0F,
    };
    std::vector<float> output(4);

    matmul_f32(lhs, rhs, output, 2, 3, 2);

    assert_vector_near(output, {58.0F, 64.0F, 139.0F, 154.0F});
}

void test_rms_norm_multiple_rows_and_weights()
{
    const std::vector<float> input {3.0F, 4.0F, -1.0F, 1.0F};
    const std::vector<float> weight {1.0F, 2.0F};
    std::vector<float> output(input.size());

    rms_norm_f32(input, weight, output, 2, 1.0F);

    const float first_inverse_sqrt = 1.0F / std::sqrt(13.5F);
    const float second_inverse_sqrt = 1.0F / std::sqrt(2.0F);
    assert_vector_near(
        output,
        {
            3.0F * first_inverse_sqrt,
            8.0F * first_inverse_sqrt,
            -second_inverse_sqrt,
            2.0F * second_inverse_sqrt,
        }
    );
}

void test_rope_half_split_rotation()
{
    const std::vector<float> input {1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> cosine {0.6F, -0.8F};
    const std::vector<float> sine {0.8F, 0.6F};
    std::vector<float> output(input.size());

    rope_f32(input, output, cosine, sine, 4);

    assert_vector_near(output, {-1.8F, -4.0F, 2.6F, -2.0F});
}

void test_sdpa_causal_mask()
{
    const std::vector<float> query {1.0F, 1.0F};
    const std::vector<float> key {1.0F, 2.0F, 3.0F};
    const std::vector<float> value {10.0F, 20.0F, 40.0F};
    std::vector<float> output(query.size());
    std::vector<float> score_workspace(key.size());

    sdpa_f32(query, key, value, output, score_workspace, 1, 0);

    const float e = std::exp(1.0F);
    const float second_query_expected = (10.0F + 20.0F * e) / (1.0F + e);
    assert_vector_near(output, {10.0F, second_query_expected});
}

void test_sdpa_query_position_offset()
{
    const std::vector<float> query {1.0F, 1.0F};
    const std::vector<float> key {1.0F, 2.0F, 3.0F};
    const std::vector<float> value {10.0F, 20.0F, 40.0F};
    std::vector<float> output(query.size());
    std::vector<float> score_workspace(key.size());

    sdpa_f32(query, key, value, output, score_workspace, 1, 1);

    const float e = std::exp(1.0F);
    const float e_squared = e * e;
    const float first_query_expected = (10.0F + 20.0F * e) / (1.0F + e);
    const float second_query_expected =
        (10.0F + 20.0F * e + 40.0F * e_squared) / (1.0F + e + e_squared);
    assert_vector_near(output, {first_query_expected, second_query_expected});
}

void test_sdpa_stable_softmax()
{
    const std::vector<float> query {1000.0F};
    const std::vector<float> key {1000.0F, 999.0F};
    const std::vector<float> value {3.0F, 7.0F};
    std::vector<float> output(query.size());
    std::vector<float> score_workspace(key.size());

    sdpa_f32(query, key, value, output, score_workspace, 1, 1);

    assert(std::isfinite(output[0]));
    assert_near(output[0], 3.0F);
}

void test_silu_mul()
{
    const std::vector<float> gate {-1.0F, 0.0F, 1.0F, -100.0F, 100.0F};
    const std::vector<float> up {2.0F, 3.0F, 4.0F, 5.0F, 2.0F};
    std::vector<float> output(gate.size());

    silu_mul_f32(gate, up, output);

    assert_near(output[0], -0.5378828F);
    assert_near(output[1], 0.0F);
    assert_near(output[2], 2.9242344F);
    assert(std::isfinite(output[3]));
    assert_near(output[3], 0.0F, 1.0e-6F);
    assert_near(output[4], 200.0F);
}

} // namespace

int main()
{
    test_matadd();
    test_matmul_rectangular_matrices();
    test_rms_norm_multiple_rows_and_weights();
    test_rope_half_split_rotation();
    test_sdpa_causal_mask();
    test_sdpa_query_position_offset();
    test_sdpa_stable_softmax();
    test_silu_mul();
}
