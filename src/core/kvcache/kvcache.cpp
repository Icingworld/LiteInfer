#include "core/kvcache/kvcache.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace liteinfer::core::kvcache
{

namespace
{

KVCacheError error(KVCacheErrorCode code, std::string_view message)
{
    return KVCacheError(code, message);
}

void validate_configuration(const KVCacheConfig & config)
{
    if (config.num_layers == 0 || config.batch_size == 0 || config.num_kv_heads == 0 ||
        config.max_seq_len == 0 || config.head_dim == 0) [[unlikely]] {
        throw std::invalid_argument("KV cache dimensions must be greater than zero");
    }

    if (config.batch_size != 1) [[unlikely]] {
        throw std::invalid_argument("KV cache currently supports batch_size == 1 only");
    }
}

std::string tensor_error_message(const tensor::TensorError & tensor_error)
{
    return std::string("Failed to allocate KV cache tensor: ") +
           std::string(tensor_error.message());
}

} // namespace

KVCache::KVCache(KVCacheConfig config)
    : config_(config)
{
    validate_configuration(config_);

    const tensor::Shape shape {
        config_.batch_size,
        config_.num_kv_heads,
        config_.max_seq_len,
        config_.head_dim,
    };

    layers_.reserve(config_.num_layers);
    for (std::size_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
        auto key = tensor::Tensor::allocate(config_.dtype, shape);
        if (!key) [[unlikely]] {
            throw std::invalid_argument(tensor_error_message(key.error()));
        }

        auto value = tensor::Tensor::allocate(config_.dtype, shape);
        if (!value) [[unlikely]] {
            throw std::invalid_argument(tensor_error_message(value.error()));
        }

        layers_.push_back(LayerStorage {std::move(*key), std::move(*value)});
    }

    written_layers_.assign(config_.num_layers, false);
}

std::expected<KVCache, KVCacheError> KVCache::create(KVCacheConfig config) noexcept
{
    try {
        return KVCache(std::move(config));
    } catch (const std::invalid_argument & exception) {
        return std::unexpected(error(KVCacheErrorCode::InvalidConfiguration, exception.what()));
    } catch (const std::length_error & exception) {
        return std::unexpected(error(KVCacheErrorCode::InvalidConfiguration, exception.what()));
    } catch (const std::bad_alloc &) {
        return std::unexpected(
            error(KVCacheErrorCode::InvalidConfiguration, "KV cache allocation failed")
        );
    }
}

std::size_t KVCache::length() const noexcept
{
    return length_;
}

std::size_t KVCache::capacity() const noexcept
{
    return config_.max_seq_len;
}

void KVCache::reset() noexcept
{
    length_ = 0;
    pending_.reset();
    std::fill(written_layers_.begin(), written_layers_.end(), false);
}

std::expected<KVCacheRegion, KVCacheError> KVCache::begin_append(std::size_t token_count)
{
    if (token_count == 0) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::InvalidOperation, "KV cache append length must be non-zero")
        );
    }

    if (pending_.has_value()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::InvalidOperation, "KV cache already has a pending append")
        );
    }

    if (length_ > capacity() || token_count > capacity() - length_) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::CapacityExceeded, "KV cache capacity is insufficient")
        );
    }

    KVCacheRegion region {
        .start = length_,
        .count = token_count,
        .end = length_ + token_count,
    };
    pending_ = region;
    std::fill(written_layers_.begin(), written_layers_.end(), false);
    return region;
}

std::expected<void, KVCacheError> KVCache::validate_region(const KVCacheRegion & region) const
{
    if (!pending_.has_value()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::InvalidOperation, "KV cache has no pending append")
        );
    }

    const auto & pending = *pending_;
    if (region.start != pending.start || region.count != pending.count || region.end != pending.end) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::InvalidOperation, "KV cache append region is not active")
        );
    }

    return {};
}

std::expected<void, KVCacheError> KVCache::validate_layer(std::size_t layer_index) const
{
    if (layer_index >= layers_.size()) [[unlikely]] {
        return std::unexpected(error(KVCacheErrorCode::InvalidLayer, "KV cache layer is invalid"));
    }
    return {};
}

std::expected<void, KVCacheError> KVCache::validate_head_major_tensors(
    const KVCacheRegion & region,
    const tensor::Tensor & key,
    const tensor::Tensor & value
) const
{
    if (key.data_type() != config_.dtype || value.data_type() != config_.dtype) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::DataTypeMismatch, "KV cache tensor data type does not match")
        );
    }

    if (!key.is_contiguous() || !value.is_contiguous()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache source tensors must be contiguous")
        );
    }

    const auto key_shape = key.shape().values();
    const auto value_shape = value.shape().values();
    const bool key_shape_matches = key_shape.size() == 4 && key_shape[0] == config_.batch_size &&
                                   key_shape[1] == config_.num_kv_heads &&
                                   key_shape[2] == region.count && key_shape[3] == config_.head_dim;
    const bool value_shape_matches =
        value_shape.size() == 4 && value_shape[0] == config_.batch_size &&
        value_shape[1] == config_.num_kv_heads && value_shape[2] == region.count &&
        value_shape[3] == config_.head_dim;
    if (!key_shape_matches || !value_shape_matches) [[unlikely]] {
        return std::unexpected(error(
            KVCacheErrorCode::ShapeMismatch,
            "KV cache tensors must have shape [batch, kv_heads, append_length, head_dim]"
        ));
    }

    if (key.numel() != value.numel()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache key/value sizes must match")
        );
    }

    return {};
}

std::expected<void, KVCacheError> KVCache::validate_token_major_tensors(
    const KVCacheRegion & region,
    const tensor::Tensor & key,
    const tensor::Tensor & value
) const
{
    if (config_.batch_size != 1) [[unlikely]] {
        return std::unexpected(error(
            KVCacheErrorCode::InvalidConfiguration,
            "Token-major KV cache writes require batch_size == 1"
        ));
    }

    if (key.data_type() != config_.dtype || value.data_type() != config_.dtype) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::DataTypeMismatch, "KV cache tensor data type does not match")
        );
    }

    if (!key.is_contiguous() || !value.is_contiguous()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache source tensors must be contiguous")
        );
    }

    const bool token_major_overflow = region.count > std::numeric_limits<std::size_t>::max() / config_.num_kv_heads;
    if (token_major_overflow) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache token-major shape overflows size_t")
        );
    }

    const auto key_shape = key.shape().values();
    const auto value_shape = value.shape().values();
    const bool key_shape_matches = key_shape.size() == 2 &&
                                   key_shape[0] == region.count * config_.num_kv_heads &&
                                   key_shape[1] == config_.head_dim;
    const bool value_shape_matches = value_shape.size() == 2 &&
                                     value_shape[0] == region.count * config_.num_kv_heads &&
                                     value_shape[1] == config_.head_dim;
    if (!key_shape_matches || !value_shape_matches) [[unlikely]] {
        return std::unexpected(error(
            KVCacheErrorCode::ShapeMismatch,
            "Token-major KV tensors must have shape [append_length * kv_heads, head_dim]"
        ));
    }

    if (key.numel() != value.numel()) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache key/value sizes must match")
        );
    }

    return {};
}

std::expected<void, KVCacheError> KVCache::mark_layer_written(std::size_t layer_index)
{
    if (written_layers_[layer_index]) [[unlikely]] {
        return std::unexpected(error(
            KVCacheErrorCode::InvalidOperation,
            "KV cache layer has already been written for this append"
        ));
    }

    written_layers_[layer_index] = true;
    return {};
}

std::expected<void, KVCacheError> KVCache::write(
    std::size_t layer_index,
    const KVCacheRegion & region,
    const tensor::Tensor & key,
    const tensor::Tensor & value
)
{
    if (auto result = validate_region(region); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    if (auto result = validate_layer(layer_index); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    if (auto result = validate_head_major_tensors(region, key, value); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    auto key_view = layers_[layer_index].key.narrow(2, region.start, region.count);
    if (!key_view) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache key storage cannot be narrowed")
        );
    }
    auto value_view = layers_[layer_index].value.narrow(2, region.start, region.count);
    if (!value_view) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache value storage cannot be narrowed")
        );
    }

    if (auto result = key_view->copy_from(key); !result) [[unlikely]] {
        return std::unexpected(error(KVCacheErrorCode::ShapeMismatch, result.error().message()));
    }
    if (auto result = value_view->copy_from(value); !result) [[unlikely]] {
        return std::unexpected(error(KVCacheErrorCode::ShapeMismatch, result.error().message()));
    }
    if (auto result = mark_layer_written(layer_index); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }

    return {};
}

std::expected<void, KVCacheError> KVCache::write_token_major(
    std::size_t layer_index,
    const KVCacheRegion & region,
    const tensor::Tensor & key,
    const tensor::Tensor & value
)
{
    if (auto result = validate_region(region); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    if (auto result = validate_layer(layer_index); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    if (auto result = validate_token_major_tensors(region, key, value); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }
    if (auto result = mark_layer_written(layer_index); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }

    const std::size_t element_size = layers_[layer_index].key.element_size();
    const std::size_t source_row_bytes = config_.head_dim * element_size;
    const std::size_t destination_head_bytes = config_.max_seq_len * source_row_bytes;
    const auto source_key = key.data();
    const auto source_value = value.data();
    auto destination_key = layers_[layer_index].key.data();
    auto destination_value = layers_[layer_index].value.data();

    for (std::size_t position = 0; position < region.count; ++position) {
        for (std::size_t head = 0; head < config_.num_kv_heads; ++head) {
            const std::size_t source_offset =
                (position * config_.num_kv_heads + head) * source_row_bytes;
            const std::size_t destination_offset =
                head * destination_head_bytes + (region.start + position) * source_row_bytes;
            std::memcpy(
                destination_key.data() + destination_offset,
                source_key.data() + source_offset,
                source_row_bytes
            );
            std::memcpy(
                destination_value.data() + destination_offset,
                source_value.data() + source_offset,
                source_row_bytes
            );
        }
    }

    return {};
}

std::expected<KVCacheView, KVCacheError>
KVCache::view(std::size_t layer_index, std::size_t visible_length) const
{
    if (auto result = validate_layer(layer_index); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }

    std::size_t maximum_visible_length = length_;
    if (pending_.has_value()) {
        maximum_visible_length = pending_->end;
    }
    if (visible_length > maximum_visible_length || visible_length > capacity()) [[unlikely]] {
        return std::unexpected(error(
            KVCacheErrorCode::InvalidOperation,
            "KV cache view exceeds the currently visible sequence length"
        ));
    }

    auto key = layers_[layer_index].key.narrow(2, 0, visible_length);
    if (!key) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache key view cannot be created")
        );
    }
    auto value = layers_[layer_index].value.narrow(2, 0, visible_length);
    if (!value) [[unlikely]] {
        return std::unexpected(
            error(KVCacheErrorCode::ShapeMismatch, "KV cache value view cannot be created")
        );
    }

    return KVCacheView {
        .key = std::move(*key),
        .value = std::move(*value),
        .length = visible_length,
    };
}

std::expected<void, KVCacheError> KVCache::commit(const KVCacheRegion & region)
{
    if (auto result = validate_region(region); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }

    for (const bool written : written_layers_) {
        if (!written) [[unlikely]] {
            return std::unexpected(error(
                KVCacheErrorCode::InvalidOperation,
                "KV cache cannot commit before every layer is written"
            ));
        }
    }

    length_ = region.end;
    pending_.reset();
    std::fill(written_layers_.begin(), written_layers_.end(), false);
    return {};
}

std::expected<void, KVCacheError> KVCache::abort(const KVCacheRegion & region) noexcept
{
    if (auto result = validate_region(region); !result) [[unlikely]] {
        return std::unexpected(std::move(result).error());
    }

    pending_.reset();
    std::fill(written_layers_.begin(), written_layers_.end(), false);
    return {};
}

std::size_t KVCache::num_layers() const noexcept
{
    return config_.num_layers;
}

std::size_t KVCache::batch_size() const noexcept
{
    return config_.batch_size;
}

std::size_t KVCache::num_kv_heads() const noexcept
{
    return config_.num_kv_heads;
}

std::size_t KVCache::head_dim() const noexcept
{
    return config_.head_dim;
}

common::data_type::DataType KVCache::data_type() const noexcept
{
    return config_.dtype;
}

} // namespace liteinfer::core::kvcache
