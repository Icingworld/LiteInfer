#pragma once

#include <cstddef>
#include <span>

namespace liteinfer::core::kernels
{

// 矩阵乘法
// lhs: 左矩阵 [m, k]
// rhs: 右矩阵 [k, n]
// output: 输出矩阵 lhs @ rhs: [m, n]
// m: 左矩阵的行数
// k: 左矩阵的列数，右矩阵的行数
// n: 右矩阵的列数
void matmul_f32(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

// 矩阵乘法，转置右矩阵
// lhs: 左矩阵 [m, k]
// rhs: 右矩阵 [n, k]
// output: 输出矩阵 lhs @ rhs^T: [m, n]
// m: 左矩阵的行数
// k: 左矩阵的列数，右矩阵的列数
// n: 右矩阵的行数，输出矩阵的列数
void matmul_transposed_rhs_f32(
    std::span<const float> lhs,
    std::span<const float> rhs,
    std::span<float> output,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

} // namespace liteinfer::core::kernels
