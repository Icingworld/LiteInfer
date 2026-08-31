#pragma once

namespace liteinfer::core::filesystem
{

// 文件访问方式
enum class FileAccess
{
    ReadOnly,
};

// 文件创建方式
enum class FileCreateMode
{
    OpenExisting,
};

// 文件系统打开文件选项
struct FileOpenOptions
{
    FileAccess access {FileAccess::ReadOnly};
    FileCreateMode create_mode {FileCreateMode::OpenExisting};
};

} // namespace liteinfer::core::filesystem
