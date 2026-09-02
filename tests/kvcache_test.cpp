#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <utility>

#include "core/common/error.hpp"
#include "core/kvcache/kvcache.hpp"
#include "core/tensor/tensor.hpp"

namespace
{

using namespace liteinfer::core;

tensor::Tensor make_float_tensor(tensor::Shape shape, std::initializer_list<float> values)
{
    auto result = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Float32,
        std::move(shape)
    );
    assert(result.has_value());

    auto data = result->data_as<float>();
    assert(data.has_value());
    assert(data->size() == values.size());
    std::size_t index = 0;
    for (const float value : values) {
        (*data)[index++] = value;
    }
    return std::move(*result);
}

tensor::Tensor materialize(const tensor::Tensor & view)
{
    auto result = tensor::Tensor::allocate(view.data_type(), view.shape());
    assert(result.has_value());
    assert(result->copy_from(view).has_value());
    return std::move(*result);
}

kvcache::KVCache make_cache()
{
    auto result = kvcache::KVCache::create(
        kvcache::KVCacheConfig {
            .num_layers = 2,
            .batch_size = 1,
            .num_kv_heads = 2,
            .max_seq_len = 4,
            .head_dim = 2,
            .dtype = liteinfer::core::common::data_type::DataType::Float32,
        }
    );
    assert(result.has_value());
    return std::move(*result);
}

void test_create_and_head_major_write()
{
    auto cache = make_cache();
    assert(cache.length() == 0);
    assert(cache.capacity() == 4);
    assert(cache.num_layers() == 2);
    assert(cache.batch_size() == 1);
    assert(cache.num_kv_heads() == 2);
    assert(cache.head_dim() == 2);
    assert(cache.data_type() == liteinfer::core::common::data_type::DataType::Float32);

    auto region = cache.begin_append(2);
    assert(region.has_value());
    assert(region->start == 0);
    assert(region->count == 2);
    assert(region->end == 2);

    auto key = make_float_tensor(
        tensor::Shape {1, 2, 2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}
    );
    auto value = make_float_tensor(
        tensor::Shape {1, 2, 2, 2},
        {11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F}
    );
    auto other_layer_key = make_float_tensor(
        tensor::Shape {1, 2, 2, 2},
        {101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 106.0F, 107.0F, 108.0F}
    );
    auto other_layer_value = make_float_tensor(
        tensor::Shape {1, 2, 2, 2},
        {111.0F, 112.0F, 113.0F, 114.0F, 115.0F, 116.0F, 117.0F, 118.0F}
    );

    assert(cache.write(0, *region, key, value).has_value());
    assert(cache.write(1, *region, other_layer_key, other_layer_value).has_value());
    assert(cache.length() == 0);

    auto view = cache.view(0, 2);
    assert(view.has_value());
    assert(view->length == 2);
    assert(!view->key.is_contiguous());
    const auto stored_shape = view->key.shape().values();
    assert(stored_shape.size() == 4);
    assert(stored_shape[0] == 1);
    assert(stored_shape[1] == 2);
    assert(stored_shape[2] == 2);
    assert(stored_shape[3] == 2);
    assert(view->key.numel() == 8);

    auto stored_key_tensor = materialize(view->key);
    auto stored_key = stored_key_tensor.data_as<float>();
    assert(stored_key.has_value());
    assert((*stored_key)[0] == 1.0F);
    assert((*stored_key)[1] == 2.0F);
    assert((*stored_key)[2] == 3.0F);
    assert((*stored_key)[3] == 4.0F);
    assert((*stored_key)[4] == 5.0F);
    assert((*stored_key)[5] == 6.0F);
    assert((*stored_key)[6] == 7.0F);
    assert((*stored_key)[7] == 8.0F);

    auto other_view = cache.view(1, 2);
    assert(other_view.has_value());
    auto other_key_tensor = materialize(other_view->key);
    auto other_key = other_key_tensor.data_as<float>();
    assert(other_key.has_value());
    assert((*other_key)[0] == 101.0F);
    assert((*other_key)[4] == 105.0F);

    assert(cache.commit(*region).has_value());
    assert(cache.length() == 2);

    cache.reset();
    assert(cache.length() == 0);
    auto reset_region = cache.begin_append(1);
    assert(reset_region.has_value());
    assert(reset_region->start == 0);
    auto reset_key =
        make_float_tensor(tensor::Shape {1, 2, 1, 2}, {201.0F, 202.0F, 203.0F, 204.0F});
    auto reset_value =
        make_float_tensor(tensor::Shape {1, 2, 1, 2}, {211.0F, 212.0F, 213.0F, 214.0F});
    assert(cache.write(0, *reset_region, reset_key, reset_value).has_value());
    assert(cache.write(1, *reset_region, reset_key, reset_value).has_value());
    assert(cache.commit(*reset_region).has_value());
    assert(cache.length() == 1);
}

void test_token_major_append_and_abort()
{
    auto cache = make_cache();

    auto first_region = cache.begin_append(2);
    assert(first_region.has_value());
    auto first_key =
        make_float_tensor(tensor::Shape {4, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto first_value =
        make_float_tensor(tensor::Shape {4, 2}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F});
    assert(cache.write_token_major(0, *first_region, first_key, first_value).has_value());
    assert(cache.write_token_major(1, *first_region, first_key, first_value).has_value());
    assert(cache.commit(*first_region).has_value());
    assert(cache.length() == 2);

    auto second_region = cache.begin_append(1);
    assert(second_region.has_value());
    assert(second_region->start == 2);
    auto second_key = make_float_tensor(tensor::Shape {2, 2}, {21.0F, 22.0F, 23.0F, 24.0F});
    auto second_value = make_float_tensor(tensor::Shape {2, 2}, {31.0F, 32.0F, 33.0F, 34.0F});
    assert(cache.write_token_major(0, *second_region, second_key, second_value).has_value());
    assert(cache.write_token_major(1, *second_region, second_key, second_value).has_value());

    auto pending_view = cache.view(0, 3);
    assert(pending_view.has_value());
    auto pending_key_tensor = materialize(pending_view->key);
    auto pending_key = pending_key_tensor.data_as<float>();
    assert(pending_key.has_value());
    assert((*pending_key)[4] == 21.0F);
    assert((*pending_key)[5] == 22.0F);
    assert((*pending_key)[10] == 23.0F);
    assert((*pending_key)[11] == 24.0F);

    assert(cache.abort(*second_region).has_value());
    assert(cache.length() == 2);
    auto reusable_region = cache.begin_append(2);
    assert(reusable_region.has_value());
    assert(reusable_region->start == 2);
    assert(cache.abort(*reusable_region).has_value());
}

void test_validation_and_transaction_rules()
{
    auto invalid_batch = kvcache::KVCache::create(
        kvcache::KVCacheConfig {
            .num_layers = 1,
            .batch_size = 2,
            .num_kv_heads = 1,
            .max_seq_len = 4,
            .head_dim = 2,
            .dtype = liteinfer::core::common::data_type::DataType::Float32,
        }
    );
    assert(!invalid_batch.has_value());
    assert(invalid_batch.error().category() == common::ErrorCategory::KVCache);

    auto cache = make_cache();
    assert(!cache.begin_append(0).has_value());
    assert(!cache.view(10, 0).has_value());

    auto region = cache.begin_append(1);
    assert(region.has_value());
    auto key = make_float_tensor(tensor::Shape {1, 2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    auto value = make_float_tensor(tensor::Shape {1, 2, 1, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    auto wrong_batch = make_float_tensor(
        tensor::Shape {2, 2, 1, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}
    );
    auto wrong_heads = make_float_tensor(tensor::Shape {1, 1, 1, 2}, {1.0F, 2.0F});
    auto wrong_sequence = make_float_tensor(
        tensor::Shape {1, 2, 2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}
    );
    auto wrong_head_dim =
        make_float_tensor(tensor::Shape {1, 2, 1, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto wrong_dtype = tensor::Tensor::allocate(
        liteinfer::core::common::data_type::DataType::Int32,
        tensor::Shape {1, 2, 1, 2}
    );
    assert(wrong_dtype.has_value());

    assert(!cache.write(2, *region, key, value).has_value());
    assert(!cache.write(0, *region, wrong_batch, wrong_batch).has_value());
    assert(!cache.write(0, *region, wrong_heads, wrong_heads).has_value());
    assert(!cache.write(0, *region, wrong_sequence, wrong_sequence).has_value());
    assert(!cache.write(0, *region, wrong_head_dim, wrong_head_dim).has_value());
    auto wrong_dtype_result = cache.write(0, *region, *wrong_dtype, *wrong_dtype);
    assert(!wrong_dtype_result.has_value());
    assert(
        wrong_dtype_result.error().code() ==
        std::to_underlying(kvcache::KVCacheErrorCode::DataTypeMismatch)
    );
    assert(!cache.begin_append(1).has_value());
    assert(!cache.view(0, 2).has_value());

    assert(cache.write(0, *region, key, value).has_value());
    assert(!cache.commit(*region).has_value());
    assert(cache.length() == 0);
    assert(cache.abort(*region).has_value());

    auto too_large = cache.begin_append(5);
    assert(!too_large.has_value());
    assert(
        too_large.error().code() == std::to_underlying(kvcache::KVCacheErrorCode::CapacityExceeded)
    );
}

} // namespace

int main()
{
    test_create_and_head_major_write();
    test_token_major_append_and_abort();
    test_validation_and_transaction_rules();
}
