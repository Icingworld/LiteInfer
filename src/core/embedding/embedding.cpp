#include "core/embedding/embedding.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace liteinfer::core::embedding
{

namespace
{

// 计算嵌入层输出，如果计算失败则返回错误
template <typename Id>
std::expected<tensor::Tensor, EmbeddingError>
lookup(const tensor::Tensor & weight, const tensor::Tensor & token_ids)
{
    const auto weight_shape = weight.shape().values();
    const std::size_t vocab_size = weight_shape[0];
    const std::size_t embedding_dim = weight_shape[1];

    auto ids = token_ids.data_as<Id>();
    if (!ids) [[unlikely]] {
        return std::unexpected(std::move(ids).error());
    }

    // 检查所有的 token_id 是否合法
    for (const Id id : *ids) {
        if constexpr (std::is_signed_v<Id>) {
            if (id < 0) [[unlikely]] {
                return std::unexpected(EmbeddingError(
                    EmbeddingErrorCode::InvalidTokenId,
                    "Token ID must not be negative"
                ));
            }
        }

        if (static_cast<std::uintmax_t>(id) >= vocab_size) [[unlikely]] {
            return std::unexpected(EmbeddingError(
                EmbeddingErrorCode::InvalidTokenId,
                "Token ID is out of vocabulary range"
            ));
        }
    }

    // 构造输出 Shape，[S, embedding_dim] 或 [B, S, embedding_dim]
    std::vector<std::size_t> output_dimensions(
        token_ids.shape().values().begin(),
        token_ids.shape().values().end()
    );
    output_dimensions.push_back(embedding_dim);

    auto output =
        tensor::Tensor::allocate(weight.data_type(), tensor::Shape(std::move(output_dimensions)));
    if (!output) [[unlikely]] {
        return std::unexpected(std::move(output).error());
    }

    // 计算每一个 token_id 所需要的字节数
    const std::size_t row_bytes = embedding_dim * weight.element_size();
    const auto source = weight.data();
    auto destination = output->data();

    for (std::size_t i = 0; i < ids->size(); ++i) {
        // 拷贝对应的权重矩阵的行到输出
        const std::size_t token_id = static_cast<std::size_t>((*ids)[i]);
        const auto * source_row = source.data() + token_id * row_bytes;
        auto * destination_row = destination.data() + i * row_bytes;
        std::memcpy(destination_row, source_row, row_bytes);
    }

    return std::move(*output);
}

} // namespace

Embedding::Embedding(tensor::Tensor weight)
    : weight_(std::move(weight))
{}

std::expected<Embedding, EmbeddingError> Embedding::create(tensor::Tensor weight)
{
    // 验证权重矩阵
    if (weight.rank() != 2) [[unlikely]] {
        return std::unexpected(EmbeddingError(
            EmbeddingErrorCode::InvalidWeightMatrix,
            "Weight matrix must be a 2D tensor"
        ));
    }
    if (*weight.shape().extent(0) == 0 || *weight.shape().extent(1) == 0) [[unlikely]] {
        return std::unexpected(EmbeddingError(
            EmbeddingErrorCode::InvalidWeightMatrix,
            "Weight matrix must be a non-empty tensor"
        ));
    }
    if (weight.data_type() != common::data_type::DataType::Float16 &&
        weight.data_type() != common::data_type::DataType::Float32 &&
        weight.data_type() != common::data_type::DataType::BFloat16) [[unlikely]] {
        return std::unexpected(EmbeddingError(
            EmbeddingErrorCode::InvalidWeightMatrix,
            "Weight matrix must be a Float16, Float32 or BFloat16 tensor"
        ));
    }
    if (!weight.is_contiguous()) [[unlikely]] {
        return std::unexpected(EmbeddingError(
            EmbeddingErrorCode::InvalidWeightMatrix,
            "Weight matrix must be a contiguous tensor"
        ));
    }

    return Embedding(std::move(weight));
}

std::expected<tensor::Tensor, EmbeddingError> Embedding::forward(
    const tensor::Tensor & token_ids
) const
{
    // 判断输入是一阶的 [S] 还是二阶的 [B, S]
    const auto input_rank = token_ids.rank();

    if (input_rank != 1 && input_rank != 2) [[unlikely]] {
        return std::unexpected(
            EmbeddingError(EmbeddingErrorCode::InvalidInputRank, "Input must be a 1D or 2D tensor")
        );
    }

    if (!token_ids.is_contiguous()) [[unlikely]] {
        return std::unexpected(
            EmbeddingError(EmbeddingErrorCode::InvalidInput, "Input must be a contiguous tensor")
        );
    }

    switch (token_ids.data_type()) {
    case common::data_type::DataType::Int32:
        return lookup<std::int32_t>(weight_, token_ids);
    case common::data_type::DataType::Int64:
        return lookup<std::int64_t>(weight_, token_ids);
    [[unlikely]] default:
        return std::unexpected(EmbeddingError(
            EmbeddingErrorCode::UnsupportedInputDataType,
            "Input must use Int32 or Int64 data type"
        ));
    }
}

std::size_t Embedding::vocab_size() const noexcept
{
    // 创建嵌入层时，权重矩阵的秩已经验证，这里认为一定不会越界
    return *weight_.shape().extent(0);
}

std::size_t Embedding::embedding_dim() const noexcept
{
    // 创建嵌入层时，权重矩阵的秩已经验证，这里认为一定不会越界
    return *weight_.shape().extent(1);
}

common::data_type::DataType Embedding::data_type() const noexcept
{
    return weight_.data_type();
}

} // namespace liteinfer::core::embedding
