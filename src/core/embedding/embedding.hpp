#pragma once

#include "core/embedding/embedding_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::embedding
{

// 嵌入层
class Embedding
{
private:
    explicit Embedding(tensor::Tensor weight);

public:
    // 创建嵌入层
    [[nodiscard]]
    static std::expected<Embedding, EmbeddingError> create(tensor::Tensor weight);

    // 计算嵌入层输出，如果计算失败则返回错误
    [[nodiscard]]
    std::expected<tensor::Tensor, EmbeddingError> forward(const tensor::Tensor & token_ids) const;

    // 获取词汇表大小
    [[nodiscard]]
    std::size_t vocab_size() const noexcept;

    // 获取嵌入维度
    [[nodiscard]]
    std::size_t embedding_dim() const noexcept;

    // 获取数据类型
    [[nodiscard]]
    tensor::DataType data_type() const noexcept;

private:
    tensor::Tensor weight_; // 权重矩阵 [vocab_size, embedding_dim]
};

} // namespace liteinfer::core::embedding
