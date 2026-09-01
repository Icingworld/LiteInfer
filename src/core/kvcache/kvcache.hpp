#pragma once

#include <cstddef>
#include <expected>
#include <optional>
#include <vector>

#include "core/kvcache/kvcache_error.hpp"
#include "core/tensor/tensor.hpp"

namespace liteinfer::core::kvcache
{

// 第一版 KV cache 配置
// 当前实现只支持单请求 batch，底层 Tensor 仍使用连续 CPU 内存
struct KVCacheConfig
{
    std::size_t num_layers;
    std::size_t batch_size;
    std::size_t num_kv_heads;
    std::size_t max_seq_len;
    std::size_t head_dim;
    tensor::DataType dtype;
};

// 一次 append 操作预留的序列区间，范围为 [start, end)
struct KVCacheRegion
{
    std::size_t start;
    std::size_t count;
    std::size_t end;
};

// KV cache 的零拷贝前缀视图
// key/value 是沿序列维度创建的 Tensor view，共享每层预分配的底层存储
struct KVCacheView
{
    tensor::Tensor key;
    tensor::Tensor value;
    std::size_t length;
};

class KVCache final
{
public:
    // 无法分配或配置非法时抛出 std::invalid_argument/std::bad_alloc
    // 推荐调用 create()，以便沿用项目的 std::expected 错误处理风格
    explicit KVCache(KVCacheConfig config);

    KVCache(const KVCache &) = delete;

    KVCache & operator=(const KVCache &) = delete;

    KVCache(KVCache &&) noexcept = default;

    KVCache & operator=(KVCache &&) noexcept = default;

    ~KVCache() = default;

public:
    // 创建 KV cache，并将构造阶段的异常转换为 KVCacheError
    [[nodiscard]]
    static std::expected<KVCache, KVCacheError> create(KVCacheConfig config) noexcept;

    // 返回当前逻辑长度
    [[nodiscard]]
    std::size_t length() const noexcept;

    // 返回当前容量
    [[nodiscard]]
    std::size_t capacity() const noexcept;

    // 重置逻辑长度，不清零已经分配的底层内存
    void reset() noexcept;

    // 预留一段新的连续序列位置。一次只允许一个未提交的 region
    [[nodiscard]]
    std::expected<KVCacheRegion, KVCacheError> begin_append(std::size_t token_count);

    // 写入 [B, Hkv, token_count, D] 布局的 K/V
    [[nodiscard]]
    std::expected<void, KVCacheError> write(
        std::size_t layer_index,
        const KVCacheRegion & region,
        const tensor::Tensor & key,
        const tensor::Tensor & value
    );

    // 写入当前 Qwen3 projection 使用的 [B, token_count, Hkv, D] 行布局
    // 这是 attention 到 cache 的零拼接适配入口；公共 cache 存储仍为 [B, Hkv, T, D]
    [[nodiscard]]
    std::expected<void, KVCacheError> write_token_major(
        std::size_t layer_index,
        const KVCacheRegion & region,
        const tensor::Tensor & key,
        const tensor::Tensor & value
    );

    // 返回有效前缀的零拷贝视图。pending append 期间允许 visible_length 到达 region.end
    [[nodiscard]]
    std::expected<KVCacheView, KVCacheError>
    view(std::size_t layer_index, std::size_t visible_length) const;

    // 所有 layer 写入完成后提交，统一推进 length
    [[nodiscard]]
    std::expected<void, KVCacheError> commit(const KVCacheRegion & region);

    // 放弃当前 append，已写入的字节会被后续 append 覆盖，但不计入有效长度
    [[nodiscard]]
    std::expected<void, KVCacheError> abort(const KVCacheRegion & region) noexcept;

    // 返回当前层数
    [[nodiscard]]
    std::size_t num_layers() const noexcept;

    // 返回当前 batch 大小
    [[nodiscard]]
    std::size_t batch_size() const noexcept;

    // 返回当前 KV head 数
    [[nodiscard]]
    std::size_t num_kv_heads() const noexcept;

    // 返回当前 head 维度
    [[nodiscard]]
    std::size_t head_dim() const noexcept;

    // 返回当前数据类型
    [[nodiscard]]
    tensor::DataType data_type() const noexcept;

private:
    struct LayerStorage
    {
        tensor::Tensor key;
        tensor::Tensor value;
    };

    [[nodiscard]]
    std::expected<void, KVCacheError> validate_region(const KVCacheRegion & region) const;

    [[nodiscard]]
    std::expected<void, KVCacheError> validate_layer(std::size_t layer_index) const;

    [[nodiscard]]
    std::expected<void, KVCacheError> validate_head_major_tensors(
        const KVCacheRegion & region,
        const tensor::Tensor & key,
        const tensor::Tensor & value
    ) const;

    [[nodiscard]]
    std::expected<void, KVCacheError> validate_token_major_tensors(
        const KVCacheRegion & region,
        const tensor::Tensor & key,
        const tensor::Tensor & value
    ) const;

    [[nodiscard]]
    std::expected<void, KVCacheError> mark_layer_written(std::size_t layer_index);

private:
    KVCacheConfig config_;
    std::vector<LayerStorage> layers_;
    std::vector<bool> written_layers_;
    std::size_t length_ = 0;
    std::optional<KVCacheRegion> pending_;
};

} // namespace liteinfer::core::kvcache
