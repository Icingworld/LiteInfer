#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/filesystem/file_handle.hpp"
#include "core/filesystem/filesystem.hpp"
#include "core/safetensors/safetensors_error.hpp"

namespace liteinfer::core::safetensors
{

// Safetensors 中 Tensor 的格式描述。
// data_begin 和 data_end 是相对于文件数据区起点的偏移。
struct TensorDescriptor
{
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_begin;
    std::uint64_t data_end;
};

// Safetensors 只读文件。
// 对象持有打开后的文件句柄，调用方提供的 Filesystem 只需要存活到 open() 返回。
class SafetensorsFile final
{
public:
    SafetensorsFile(const SafetensorsFile &) = delete;

    SafetensorsFile & operator=(const SafetensorsFile &) = delete;

    SafetensorsFile(SafetensorsFile &&) noexcept;

    SafetensorsFile & operator=(SafetensorsFile &&) noexcept;

    ~SafetensorsFile();

public:
    // 打开并完整校验一个 Safetensors 文件。
    [[nodiscard]]
    static std::expected<SafetensorsFile, SafetensorsError>
    open(filesystem::Filesystem & filesystem, const std::filesystem::path & path);

    // 获取所有 Tensor 名称，名称按字典序返回。
    [[nodiscard]]
    std::vector<std::string> tensor_names() const;

    // 获取指定 Tensor 的格式描述。
    [[nodiscard]]
    std::expected<TensorDescriptor, SafetensorsError> tensor_info(std::string_view name) const;

    // 获取文件元数据。返回的引用在当前对象存活期间有效。
    [[nodiscard]]
    const std::map<std::string, std::string> & metadata() const noexcept;

    // 读取指定 Tensor 的原始字节。返回拥有数据的副本。
    [[nodiscard]]
    std::expected<std::vector<std::byte>, SafetensorsError> read_tensor(std::string_view name);

private:
    SafetensorsFile(
        filesystem::FileHandle file,
        std::uint64_t file_size,
        std::uint64_t data_region_begin,
        std::map<std::string, TensorDescriptor> tensors,
        std::map<std::string, std::string> metadata
    ) noexcept;

private:
    filesystem::FileHandle file_;
    std::uint64_t file_size_;
    std::uint64_t data_region_begin_;
    std::map<std::string, TensorDescriptor> tensors_;
    std::map<std::string, std::string> metadata_;
};

} // namespace liteinfer::core::safetensors
