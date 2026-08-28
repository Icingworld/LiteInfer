#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace liteinfer::core::common
{

// 错误码分类
enum class ErrorCategory : std::uint8_t
{
    Unknown = 0,
    Tensor = 1,
    Embedding = 2,
    Layer = 3,
};

// 通用错误类型模板
// 当具体的错误类型 E 被定义时，必须特化 ErrorTraits 模板类，并提供 category 静态成员变量
// 例如:
// template <>
// struct ErrorTraits<MyError>
// {
//     static constexpr ErrorCategory category = ErrorCategory::MyError;
// };
template <typename E>
struct ErrorTraits;

// 错误类型概念
// 错误类型必须是一个枚举类型，且底层类型为 std::uint8_t
// 错误类型必须有一个 category 静态成员变量，且类型为 ErrorCategory
template <typename E>
concept ErrorType =
    std::is_enum_v<E> && std::same_as<std::underlying_type_t<E>, std::uint8_t> && requires {
        { ErrorTraits<E>::category } -> std::same_as<const ErrorCategory &>;
    };

// 错误类型
// 所有模块的错误都使用该类型构造，允许在模块内部使用 using 定义别名
// 总体上，错误应当在产生后原样上浮到顶层，而不是在模块内部映射转换
class Error
{
public:
    template <ErrorType E>
    explicit Error(E error, std::string_view message)
        : category_(ErrorTraits<E>::category)
        , code_(std::to_underlying(error))
        , message_(message)
    {}

    Error(const Error &) = delete;

    Error & operator=(const Error &) = delete;

    Error(Error &&) noexcept = default;

    Error & operator=(Error &&) noexcept = default;

public:
    // 获取错误分类
    [[nodiscard]]
    ErrorCategory category() const noexcept
    {
        return category_;
    }

    // 获取错误码
    [[nodiscard]]
    std::uint8_t code() const noexcept
    {
        return code_;
    }

    // 获取错误信息
    [[nodiscard]]
    std::string_view message() const noexcept
    {
        return message_;
    }

private:
    ErrorCategory category_;
    std::uint8_t code_; // 模块内部定义的错误码
    std::string message_;
};

} // namespace liteinfer::core::common
