#include "core/tokenizer/qwen3/qwen3_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/io/file_byte_reader.hpp"
#include "core/json/json.hpp"

namespace liteinfer::core::tokenizer::qwen3
{

namespace
{

constexpr std::string_view TOKENIZER_FILE_NAME = "tokenizer.json";
constexpr std::string_view TOKENIZER_CONFIG_FILE_NAME = "tokenizer_config.json";
constexpr std::uint64_t TOKENIZER_FILE_SIZE_LIMIT = 64ULL * 1024ULL * 1024ULL;
constexpr std::string_view PRETOKENIZE_REGEX =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

TokenizerError invalid_configuration(std::string_view message)
{
    return TokenizerError(TokenizerErrorCode::InvalidConfiguration, message);
}

TokenizerError invalid_input(std::string_view message)
{
    return TokenizerError(TokenizerErrorCode::InvalidInput, message);
}

TokenizerError unsupported_feature(std::string_view message)
{
    return TokenizerError(TokenizerErrorCode::UnsupportedFeature, message);
}

TokenizerError invalid_field(std::string_view field, std::string_view reason)
{
    std::string message;
    message.reserve(field.size() + reason.size() + 32);
    message += "Qwen3 tokenizer field '";
    message += field;
    message += "' ";
    message += reason;
    return invalid_configuration(message);
}

bool has_string_value(
    const json::Document & object,
    std::string_view field,
    std::string_view expected
)
{
    const std::string key(field);
    return object.contains(key) && object.at(key).is_string() &&
           object.at(key).get<std::string>() == expected;
}

struct MergePair
{
    std::string first;
    std::string second;

    bool operator==(const MergePair & other) const noexcept
    {
        return first == other.first && second == other.second;
    }
};

struct MergePairHash
{
    std::size_t operator()(const MergePair & value) const noexcept
    {
        const std::size_t first_hash = std::hash<std::string> {}(value.first);
        const std::size_t second_hash = std::hash<std::string> {}(value.second);
        return first_hash ^ (second_hash + static_cast<std::size_t>(0x9e3779b9) +
                             (first_hash << 6U) + (first_hash >> 2U));
    }
};

struct AddedToken
{
    common::TokenId id;
    std::string content;
    bool special;
};

struct ByteLevelCodec
{
    std::array<std::string, 256> byte_symbols;
    std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte;
};
} // namespace

struct Qwen3Tokenizer::Impl
{
    std::size_t base_vocab_size {0};
    std::unordered_map<std::string, common::TokenId> token_to_id;
    std::vector<std::string> id_to_token;
    std::vector<bool> id_present;
    std::vector<bool> id_is_added;
    std::vector<bool> id_is_special;
    std::unordered_map<MergePair, std::size_t, MergePairHash> merge_ranks;
    std::vector<AddedToken> added_tokens;
    std::optional<common::TokenId> bos_token_id;
    std::optional<common::TokenId> eos_token_id;
    std::optional<common::TokenId> pad_token_id;
    ByteLevelCodec byte_level;
};

namespace
{

void append_utf8(std::string & output, std::uint32_t codepoint)
{
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
}

bool decode_utf8_at(
    std::string_view input,
    std::size_t offset,
    std::uint32_t & codepoint,
    std::size_t & width
) noexcept
{
    if (offset >= input.size()) {
        return false;
    }

    const auto byte_at = [&](std::size_t index) {
        return static_cast<std::uint8_t>(static_cast<unsigned char>(input[index]));
    };

    const std::uint8_t first = byte_at(offset);
    if (first <= 0x7fU) {
        codepoint = first;
        width = 1;
        return true;
    }

    std::size_t expected_width = 0;
    std::uint32_t value = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
        expected_width = 2;
        value = first & 0x1fU;
        minimum = 0x80U;
    } else if (first >= 0xe0U && first <= 0xefU) {
        expected_width = 3;
        value = first & 0x0fU;
        minimum = 0x800U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
        expected_width = 4;
        value = first & 0x07U;
        minimum = 0x10000U;
    } else {
        return false;
    }

    if (offset + expected_width > input.size()) {
        return false;
    }

    for (std::size_t index = 1; index < expected_width; ++index) {
        const auto byte = byte_at(offset + index);
        if ((byte & 0xc0U) != 0x80U) {
            return false;
        }
        value = (value << 6U) | (byte & 0x3fU);
    }

    if (value < minimum || value > 0x10ffffU || (value >= 0xd800U && value <= 0xdfffU)) {
        return false;
    }

    codepoint = value;
    width = expected_width;
    return true;
}

std::expected<std::vector<std::uint32_t>, TokenizerError> decode_utf8(std::string_view input)
{
    std::vector<std::uint32_t> codepoints;
    codepoints.reserve(input.size());

    std::size_t offset = 0;
    while (offset < input.size()) {
        std::uint32_t codepoint = 0;
        std::size_t width = 0;
        if (!decode_utf8_at(input, offset, codepoint, width)) [[unlikely]] {
            return std::unexpected(invalid_input("Tokenizer input must be valid UTF-8"));
        }
        codepoints.push_back(codepoint);
        offset += width;
    }

    return codepoints;
}

std::string decode_utf8_with_replacement(std::string_view input)
{
    constexpr std::string_view REPLACEMENT = "\xEF\xBF\xBD";
    std::string output;
    output.reserve(input.size());

    std::size_t offset = 0;
    while (offset < input.size()) {
        std::uint32_t codepoint = 0;
        std::size_t width = 0;
        if (!decode_utf8_at(input, offset, codepoint, width)) {
            output += REPLACEMENT;
            ++offset;
        } else {
            output.append(input.substr(offset, width));
            offset += width;
        }
    }

    return output;
}

ByteLevelCodec make_byte_level_codec()
{
    ByteLevelCodec codec;
    std::array<std::uint32_t, 256> byte_to_codepoint {};
    std::array<bool, 256> included {};

    for (std::uint32_t byte = 0x21U; byte <= 0x7eU; ++byte) {
        included[byte] = true;
        byte_to_codepoint[byte] = byte;
    }
    for (std::uint32_t byte = 0xa1U; byte <= 0xacU; ++byte) {
        included[byte] = true;
        byte_to_codepoint[byte] = byte;
    }
    for (std::uint32_t byte = 0xaeU; byte <= 0xffU; ++byte) {
        included[byte] = true;
        byte_to_codepoint[byte] = byte;
    }

    std::uint32_t extra = 0;
    for (std::uint32_t byte = 0; byte < 256U; ++byte) {
        if (!included[byte]) {
            byte_to_codepoint[byte] = 256U + extra;
            ++extra;
        }
    }

    codec.unicode_to_byte.reserve(256);
    for (std::uint32_t byte = 0; byte < 256U; ++byte) {
        append_utf8(codec.byte_symbols[byte], byte_to_codepoint[byte]);
        codec.unicode_to_byte.emplace(byte_to_codepoint[byte], static_cast<std::uint8_t>(byte));
    }

    return codec;
}

bool is_number(std::uint32_t codepoint) noexcept
{
    return (codepoint >= '0' && codepoint <= '9') ||
           (codepoint >= 0x0660U && codepoint <= 0x0669U) ||
           (codepoint >= 0x06f0U && codepoint <= 0x06f9U) ||
           (codepoint >= 0x0966U && codepoint <= 0x096fU) ||
           (codepoint >= 0x09e6U && codepoint <= 0x09efU) ||
           (codepoint >= 0x0a66U && codepoint <= 0x0a6fU) ||
           (codepoint >= 0x0ae6U && codepoint <= 0x0aefU) ||
           (codepoint >= 0x0b66U && codepoint <= 0x0b6fU) ||
           (codepoint >= 0x0be6U && codepoint <= 0x0befU) ||
           (codepoint >= 0x0c66U && codepoint <= 0x0c6fU) ||
           (codepoint >= 0x0ce6U && codepoint <= 0x0cefU) ||
           (codepoint >= 0x0d66U && codepoint <= 0x0d6fU) ||
           (codepoint >= 0x0e50U && codepoint <= 0x0e59U) ||
           (codepoint >= 0x0ed0U && codepoint <= 0x0ed9U) ||
           (codepoint >= 0x0f20U && codepoint <= 0x0f29U) ||
           (codepoint >= 0x1040U && codepoint <= 0x1049U) ||
           (codepoint >= 0x17e0U && codepoint <= 0x17e9U) ||
           (codepoint >= 0xff10U && codepoint <= 0xff19U) ||
           (codepoint >= 0x1d7ceU && codepoint <= 0x1d7ffU) ||
           (codepoint >= 0x2160U && codepoint <= 0x2188U);
}

bool is_letter(std::uint32_t codepoint) noexcept
{
    if (is_number(codepoint)) {
        return false;
    }

    if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z')) {
        return true;
    }

    return (codepoint >= 0x00c0U && codepoint <= 0x02afU && codepoint != 0x00d7U &&
            codepoint != 0x00f7U) ||
           (codepoint >= 0x0370U && codepoint <= 0x052fU) ||
           (codepoint >= 0x0531U && codepoint <= 0x058fU) ||
           (codepoint >= 0x05d0U && codepoint <= 0x05eaU) ||
           (codepoint >= 0x0620U && codepoint <= 0x06ffU) ||
           (codepoint >= 0x0710U && codepoint <= 0x072fU) ||
           (codepoint >= 0x0780U && codepoint <= 0x07b1U) ||
           (codepoint >= 0x0900U && codepoint <= 0x0dffU) ||
           (codepoint >= 0x0e01U && codepoint <= 0x0e5bU) ||
           (codepoint >= 0x0f40U && codepoint <= 0x0fffU) ||
           (codepoint >= 0x1000U && codepoint <= 0x137fU) ||
           (codepoint >= 0x13a0U && codepoint <= 0x13ffU) ||
           (codepoint >= 0x1401U && codepoint <= 0x167fU) ||
           (codepoint >= 0x1681U && codepoint <= 0x169fU) ||
           (codepoint >= 0x16a0U && codepoint <= 0x16ffU) ||
           (codepoint >= 0x1700U && codepoint <= 0x177fU) ||
           (codepoint >= 0x1780U && codepoint <= 0x17ffU) ||
           (codepoint >= 0x1e00U && codepoint <= 0x1effU) ||
           (codepoint >= 0x3041U && codepoint <= 0x30ffU) ||
           (codepoint >= 0x3105U && codepoint <= 0x318fU) ||
           (codepoint >= 0x3400U && codepoint <= 0x4dbfU) ||
           (codepoint >= 0x4e00U && codepoint <= 0x9fffU) ||
           (codepoint >= 0xa000U && codepoint <= 0xa4cfU) ||
           (codepoint >= 0xac00U && codepoint <= 0xd7ffU) ||
           (codepoint >= 0xf900U && codepoint <= 0xfaffU) ||
           (codepoint >= 0xff21U && codepoint <= 0xff3aU) ||
           (codepoint >= 0xff41U && codepoint <= 0xff5aU) ||
           (codepoint >= 0x20000U && codepoint <= 0x3134fU);
}

bool is_whitespace(std::uint32_t codepoint) noexcept
{
    return (codepoint >= 0x09U && codepoint <= 0x0dU) || codepoint == 0x20U || codepoint == 0x85U ||
           codepoint == 0xa0U || codepoint == 0x1680U ||
           (codepoint >= 0x2000U && codepoint <= 0x200aU) || codepoint == 0x2028U ||
           codepoint == 0x2029U || codepoint == 0x202fU || codepoint == 0x205fU ||
           codepoint == 0x3000U;
}

bool is_newline(std::uint32_t codepoint) noexcept
{
    return codepoint == 0x0aU || codepoint == 0x0dU;
}

bool matches_case_insensitive(std::uint32_t codepoint, char lowercase) noexcept
{
    return codepoint == static_cast<std::uint32_t>(lowercase) ||
           codepoint == static_cast<std::uint32_t>(lowercase - 'a' + 'A');
}

std::size_t
contraction_length(const std::vector<std::uint32_t> & codepoints, std::size_t position) noexcept
{
    if (position >= codepoints.size() || codepoints[position] != '\'') {
        return 0;
    }

    constexpr std::array<std::string_view, 7>
        SUFFIXES {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (const auto suffix : SUFFIXES) {
        if (position + suffix.size() > codepoints.size()) {
            continue;
        }

        bool matches = true;
        for (std::size_t offset = 1; offset < suffix.size(); ++offset) {
            if (!matches_case_insensitive(codepoints[position + offset], suffix[offset])) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return suffix.size();
        }
    }
    return 0;
}

std::size_t
next_pretoken_end(const std::vector<std::uint32_t> & codepoints, std::size_t position) noexcept
{
    const std::size_t size = codepoints.size();

    if (const auto length = contraction_length(codepoints, position); length != 0) {
        return position + length;
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    std::size_t letters_begin = position;
    if (letters_begin < size && !is_newline(codepoints[letters_begin]) &&
        !is_letter(codepoints[letters_begin]) && !is_number(codepoints[letters_begin])) {
        ++letters_begin;
    }
    if (letters_begin < size && is_letter(codepoints[letters_begin])) {
        std::size_t end = letters_begin + 1;
        while (end < size && is_letter(codepoints[end])) {
            ++end;
        }
        return end;
    }

    // \p{N}
    if (is_number(codepoints[position])) {
        return position + 1;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    std::size_t other_begin = position;
    if (codepoints[other_begin] == ' ') {
        ++other_begin;
    }
    if (other_begin < size && !is_whitespace(codepoints[other_begin]) &&
        !is_letter(codepoints[other_begin]) && !is_number(codepoints[other_begin])) {
        std::size_t end = other_begin + 1;
        while (end < size && !is_whitespace(codepoints[end]) && !is_letter(codepoints[end]) &&
               !is_number(codepoints[end])) {
            ++end;
        }
        while (end < size && is_newline(codepoints[end])) {
            ++end;
        }
        return end;
    }

    // \s*[\r\n]+
    std::size_t before_newline = position;
    while (before_newline < size && is_whitespace(codepoints[before_newline]) &&
           !is_newline(codepoints[before_newline])) {
        ++before_newline;
    }
    if (before_newline < size && is_newline(codepoints[before_newline])) {
        std::size_t end = before_newline;
        while (end < size && is_newline(codepoints[end])) {
            ++end;
        }
        return end;
    }

    // \s+
    if (is_whitespace(codepoints[position])) {
        std::size_t end = position + 1;
        while (end < size && is_whitespace(codepoints[end])) {
            ++end;
        }
        return end;
    }

    // All valid code points should have matched one of the branches.
    return position + 1;
}

struct PretokenSpan
{
    std::size_t begin;
    std::size_t end;
};

std::vector<PretokenSpan> split_pretokens(const std::vector<std::uint32_t> & codepoints)
{
    std::vector<PretokenSpan> result;
    std::size_t position = 0;
    while (position < codepoints.size()) {
        const std::size_t end = next_pretoken_end(codepoints, position);
        result.push_back(PretokenSpan {.begin = position, .end = end});
        position = end;
    }
    return result;
}

std::expected<json::Document, TokenizerError>
load_json_document(filesystem::Filesystem & filesystem, const std::filesystem::path & path)
{
    auto file_result = filesystem.open(path);
    if (!file_result) [[unlikely]] {
        return std::unexpected(std::move(file_result).error());
    }
    auto file = std::move(*file_result);

    auto size_result = file.size();
    if (!size_result) [[unlikely]] {
        return std::unexpected(std::move(size_result).error());
    }
    if (*size_result > TOKENIZER_FILE_SIZE_LIMIT ||
        *size_result > std::numeric_limits<std::size_t>::max()) [[unlikely]] {
        return std::unexpected(invalid_configuration("Tokenizer JSON file is too large"));
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(*size_result));
    io::FileByteReader reader(file);
    auto read_result = reader.read_exact(bytes);
    if (!read_result) [[unlikely]] {
        return std::unexpected(std::move(read_result).error());
    }

    std::string text;
    if (!bytes.empty()) {
        text.assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }

    try {
        return json::Document::parse(text);
    } catch (const nlohmann::json::exception &) {
        return std::unexpected(invalid_configuration("Tokenizer file is not valid JSON"));
    }
}

std::expected<common::TokenId, TokenizerError>
parse_token_id(const json::Document & value, std::string_view field)
{
    std::uint64_t parsed = 0;
    if (value.is_number_unsigned()) {
        parsed = value.get<std::uint64_t>();
    } else if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0) [[unlikely]] {
            return std::unexpected(invalid_field(field, "must be a non-negative integer"));
        }
        parsed = static_cast<std::uint64_t>(signed_value);
    } else [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a non-negative integer"));
    }

    if (parsed > std::numeric_limits<common::TokenId>::max()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "does not fit in TokenId"));
    }
    return static_cast<common::TokenId>(parsed);
}

std::expected<common::TokenId, TokenizerError>
parse_token_id_text(std::string_view value, std::string_view field)
{
    std::uint64_t parsed = 0;
    const auto begin = value.data();
    const auto end = begin + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc {} || result.ptr != end ||
        parsed > std::numeric_limits<common::TokenId>::max()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a valid TokenId"));
    }
    return static_cast<common::TokenId>(parsed);
}

std::expected<AddedToken, TokenizerError>
parse_added_token(const json::Document & value, common::TokenId id, std::string_view field)
{
    if (!value.is_object()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be an object"));
    }
    if (!value.contains("content") || !value.at("content").is_string()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "content must be a string"));
    }

    const auto content = value.at("content").get<std::string>();
    if (content.empty()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "content must not be empty"));
    }

    const auto read_false_flag = [&](std::string_view name) -> std::expected<void, TokenizerError> {
        const std::string key(name);
        if (!value.contains(key)) {
            return {};
        }
        if (!value.at(key).is_boolean() || value.at(key).get<bool>()) [[unlikely]] {
            return std::unexpected(unsupported_feature("Qwen3 added token flags must be false"));
        }
        return {};
    };

    for (const auto flag : {"single_word", "lstrip", "rstrip", "normalized"}) {
        auto result = read_false_flag(flag);
        if (!result) [[unlikely]] {
            return std::unexpected(std::move(result).error());
        }
    }

    bool special = false;
    if (value.contains("special")) {
        if (!value.at("special").is_boolean()) [[unlikely]] {
            return std::unexpected(invalid_field(field, "special must be a boolean"));
        }
        special = value.at("special").get<bool>();
    }

    return AddedToken {
        .id = id,
        .content = content,
        .special = special,
    };
}

std::expected<void, TokenizerError> add_added_token(Qwen3Tokenizer::Impl & impl, AddedToken token)
{
    if (token.id < impl.base_vocab_size) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3 added token ID overlaps the base vocabulary")
        );
    }

    const auto existing_token = impl.token_to_id.find(token.content);
    if (existing_token != impl.token_to_id.end() && existing_token->second != token.id)
        [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3 tokenizer maps one token string to multiple IDs")
        );
    }

    if (static_cast<std::size_t>(token.id) >= impl.id_to_token.size()) {
        const auto new_size = static_cast<std::size_t>(token.id) + 1;
        if (new_size > 16U * 1024U * 1024U) [[unlikely]] {
            return std::unexpected(invalid_configuration("Qwen3 tokenizer ID table is too large"));
        }
        impl.id_to_token.resize(new_size);
        impl.id_present.resize(new_size, false);
        impl.id_is_added.resize(new_size, false);
        impl.id_is_special.resize(new_size, false);
    }

    const auto index = static_cast<std::size_t>(token.id);
    if (impl.id_present[index] && impl.id_to_token[index] != token.content) [[unlikely]] {
        return std::unexpected(
            invalid_configuration("Qwen3 tokenizer maps one ID to multiple token strings")
        );
    }

    impl.id_to_token[index] = token.content;
    impl.id_present[index] = true;
    impl.id_is_added[index] = true;
    impl.id_is_special[index] = token.special;
    impl.token_to_id[token.content] = token.id;

    const auto existing_added = std::find_if(
        impl.added_tokens.begin(),
        impl.added_tokens.end(),
        [id = token.id](const AddedToken & value) {
            return value.id == id;
        }
    );
    if (existing_added == impl.added_tokens.end()) {
        impl.added_tokens.push_back(std::move(token));
    } else {
        *existing_added = std::move(token);
    }
    return {};
}

std::expected<std::optional<common::TokenId>, TokenizerError> read_special_token_id(
    const json::Document & config,
    std::string_view field,
    const Qwen3Tokenizer::Impl & impl
)
{
    const std::string key(field);
    if (!config.contains(key) || config.at(key).is_null()) {
        return std::nullopt;
    }
    if (!config.at(key).is_string()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "must be a string or null"));
    }

    const auto token = config.at(key).get<std::string>();
    const auto found = impl.token_to_id.find(token);
    if (found == impl.token_to_id.end()) [[unlikely]] {
        return std::unexpected(invalid_field(field, "refers to an unknown token"));
    }
    return found->second;
}

std::expected<void, TokenizerError> validate_pipeline(const json::Document & document)
{
    if (!document.is_object()) [[unlikely]] {
        return std::unexpected(invalid_configuration("Tokenizer root must be an object"));
    }

    if (!document.contains("normalizer") || !document.at("normalizer").is_object() ||
        !has_string_value(document.at("normalizer"), "type", "NFC")) [[unlikely]] {
        return std::unexpected(unsupported_feature("Qwen3 tokenizer must use NFC normalizer"));
    }

    if (!document.contains("pre_tokenizer") || !document.at("pre_tokenizer").is_object())
        [[unlikely]] {
        return std::unexpected(invalid_configuration("Tokenizer pre_tokenizer is missing"));
    }

    const auto & pre_tokenizer = document.at("pre_tokenizer");
    if (!has_string_value(pre_tokenizer, "type", "Sequence") ||
        !pre_tokenizer.contains("pretokenizers") || !pre_tokenizer.at("pretokenizers").is_array() ||
        pre_tokenizer.at("pretokenizers").size() != 2) [[unlikely]] {
        return std::unexpected(
            unsupported_feature("Qwen3 tokenizer pre_tokenizer is not the supported sequence")
        );
    }

    const auto & split = pre_tokenizer.at("pretokenizers").at(0);
    if (!split.is_object() || !has_string_value(split, "type", "Split") ||
        !has_string_value(split, "behavior", "Isolated") || !split.contains("pattern") ||
        !split.at("pattern").is_object() ||
        !has_string_value(split.at("pattern"), "Regex", PRETOKENIZE_REGEX)) [[unlikely]] {
        return std::unexpected(
            unsupported_feature("Qwen3 tokenizer Split pre-tokenizer is not supported")
        );
    }

    const auto & byte_level = pre_tokenizer.at("pretokenizers").at(1);
    if (!byte_level.is_object() || !has_string_value(byte_level, "type", "ByteLevel") ||
        byte_level.value("add_prefix_space", true) || byte_level.value("use_regex", true))
        [[unlikely]] {
        return std::unexpected(
            unsupported_feature("Qwen3 tokenizer ByteLevel pre-tokenizer is not supported")
        );
    }

    for (const auto field : {"post_processor", "decoder"}) {
        if (!document.contains(field) || !document.at(field).is_object() ||
            !has_string_value(document.at(field), "type", "ByteLevel")) [[unlikely]] {
            return std::unexpected(
                unsupported_feature("Qwen3 tokenizer must use ByteLevel post-processing")
            );
        }
    }

    return {};
}

std::expected<std::vector<std::string>, TokenizerError>
apply_bpe(const Qwen3Tokenizer::Impl & impl, const std::vector<std::string> & symbols)
{
    std::vector<std::string> result = symbols;
    while (result.size() > 1) {
        std::size_t best_index = result.size();
        std::size_t best_rank = std::numeric_limits<std::size_t>::max();

        for (std::size_t index = 0; index + 1 < result.size(); ++index) {
            const MergePair pair {.first = result[index], .second = result[index + 1]};
            const auto merge = impl.merge_ranks.find(pair);
            if (merge != impl.merge_ranks.end() && merge->second < best_rank) {
                best_index = index;
                best_rank = merge->second;
            }
        }

        if (best_index == result.size()) {
            break;
        }

        result[best_index] += result[best_index + 1];
        result.erase(result.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    return result;
}

std::expected<void, TokenizerError> encode_plain_text(
    const Qwen3Tokenizer::Impl & impl,
    std::string_view text,
    std::vector<common::TokenId> & output
)
{
    auto codepoints = decode_utf8(text);
    if (!codepoints) [[unlikely]] {
        return std::unexpected(std::move(codepoints).error());
    }

    for (const auto pretoken : split_pretokens(*codepoints)) {
        std::vector<std::string> symbols;
        for (std::size_t index = pretoken.begin; index < pretoken.end; ++index) {
            std::string utf8_codepoint;
            append_utf8(utf8_codepoint, (*codepoints)[index]);
            for (const auto byte : utf8_codepoint) {
                symbols.push_back(
                    impl.byte_level
                        .byte_symbols[static_cast<std::uint8_t>(static_cast<unsigned char>(byte))]
                );
            }
        }

        auto merged = apply_bpe(impl, symbols);
        if (!merged) [[unlikely]] {
            return std::unexpected(std::move(merged).error());
        }

        for (const auto & token : *merged) {
            const auto found = impl.token_to_id.find(token);
            if (found == impl.token_to_id.end()) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 BPE produced a token missing from vocabulary")
                );
            }
            output.push_back(found->second);
        }
    }

    return {};
}

std::expected<void, TokenizerError> append_decoded_token(
    const Qwen3Tokenizer::Impl & impl,
    common::TokenId token_id,
    std::string & byte_buffer,
    std::string & output,
    bool skip_special_tokens
)
{
    const auto index = static_cast<std::size_t>(token_id);
    if (index >= impl.id_to_token.size() || !impl.id_present[index]) {
        // Qwen tokenizer 保留了一段没有对应字符串的模型词表 ID，官方 decode 会忽略它们。
        return {};
    }

    if (impl.id_is_added[index]) {
        if (!byte_buffer.empty()) {
            output += decode_utf8_with_replacement(byte_buffer);
            byte_buffer.clear();
        }
        if (!skip_special_tokens || !impl.id_is_special[index]) {
            output += impl.id_to_token[index];
        }
        return {};
    }

    const auto token = impl.id_to_token[index];
    std::size_t offset = 0;
    while (offset < token.size()) {
        std::uint32_t codepoint = 0;
        std::size_t width = 0;
        if (!decode_utf8_at(token, offset, codepoint, width)) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 vocabulary contains invalid UTF-8")
            );
        }

        const auto byte = impl.byte_level.unicode_to_byte.find(codepoint);
        if (byte == impl.byte_level.unicode_to_byte.end()) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 vocabulary contains a non-ByteLevel token")
            );
        }
        byte_buffer.push_back(static_cast<char>(byte->second));
        offset += width;
    }
    return {};
}

} // namespace

Qwen3Tokenizer::Qwen3Tokenizer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{}

Qwen3Tokenizer::Qwen3Tokenizer(Qwen3Tokenizer &&) noexcept = default;

Qwen3Tokenizer & Qwen3Tokenizer::operator=(Qwen3Tokenizer &&) noexcept = default;

Qwen3Tokenizer::~Qwen3Tokenizer() = default;

std::expected<Qwen3Tokenizer, TokenizerError> Qwen3Tokenizer::load(
    filesystem::Filesystem & filesystem,
    const std::filesystem::path & model_directory
)
{
    auto tokenizer_document = load_json_document(filesystem, model_directory / TOKENIZER_FILE_NAME);
    if (!tokenizer_document) [[unlikely]] {
        return std::unexpected(std::move(tokenizer_document).error());
    }

    auto config_document =
        load_json_document(filesystem, model_directory / TOKENIZER_CONFIG_FILE_NAME);
    if (!config_document) [[unlikely]] {
        return std::unexpected(std::move(config_document).error());
    }

    try {
        auto pipeline_result = validate_pipeline(*tokenizer_document);
        if (!pipeline_result) [[unlikely]] {
            return std::unexpected(std::move(pipeline_result).error());
        }

        const auto & model = tokenizer_document->at("model");
        if (!model.is_object() || !has_string_value(model, "type", "BPE") ||
            model.value("byte_fallback", true)) [[unlikely]] {
            return std::unexpected(
                unsupported_feature("Qwen3 tokenizer must use BPE without byte fallback")
            );
        }

        const auto & vocab = model.at("vocab");
        if (!vocab.is_object() || vocab.empty()) [[unlikely]] {
            return std::unexpected(invalid_configuration("Qwen3 tokenizer vocabulary is empty"));
        }
        if (vocab.size() > 16U * 1024U * 1024U) [[unlikely]] {
            return std::unexpected(
                invalid_configuration("Qwen3 tokenizer vocabulary is too large")
            );
        }

        auto impl = std::make_unique<Impl>();
        impl->base_vocab_size = vocab.size();
        impl->token_to_id.reserve(vocab.size() + 64U);
        impl->id_to_token.resize(vocab.size());
        impl->id_present.resize(vocab.size(), false);
        impl->id_is_added.resize(vocab.size(), false);
        impl->id_is_special.resize(vocab.size(), false);

        for (const auto & [token, value] : vocab.items()) {
            auto token_id = parse_token_id(value, "model.vocab");
            if (!token_id) [[unlikely]] {
                return std::unexpected(std::move(token_id).error());
            }
            const auto index = static_cast<std::size_t>(*token_id);
            if (index >= impl->base_vocab_size) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 base vocabulary IDs must be contiguous")
                );
            }
            if (impl->id_present[index]) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 base vocabulary contains duplicate IDs")
                );
            }
            impl->id_to_token[index] = token;
            impl->id_present[index] = true;

            const auto [found, inserted] = impl->token_to_id.emplace(token, *token_id);
            if (!inserted && found->second != *token_id) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 vocabulary maps one token to multiple IDs")
                );
            }
        }

        for (const bool present : impl->id_present) {
            if (!present) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 base vocabulary IDs must be contiguous")
                );
            }
        }

        const auto & merges = model.at("merges");
        if (!merges.is_array()) [[unlikely]] {
            return std::unexpected(invalid_configuration("Qwen3 BPE merges must be an array"));
        }
        impl->merge_ranks.reserve(merges.size());
        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            const auto & value = merges.at(rank);
            if (!value.is_string()) [[unlikely]] {
                return std::unexpected(invalid_configuration("Qwen3 BPE merge must be a string"));
            }
            const auto merge = value.get<std::string>();
            const auto separator = merge.find(' ');
            if (separator == std::string::npos || separator == 0 || separator + 1 >= merge.size())
                [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 BPE merge must contain two token strings")
                );
            }

            MergePair pair {
                .first = merge.substr(0, separator),
                .second = merge.substr(separator + 1),
            };
            const auto [_, inserted] = impl->merge_ranks.emplace(std::move(pair), rank);
            if (!inserted) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 BPE contains duplicate merge rules")
                );
            }
        }

        if (tokenizer_document->contains("added_tokens")) {
            const auto & added_tokens = tokenizer_document->at("added_tokens");
            if (!added_tokens.is_array()) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("Qwen3 added_tokens must be an array")
                );
            }
            for (std::size_t index = 0; index < added_tokens.size(); ++index) {
                const auto & value = added_tokens.at(index);
                auto id = parse_token_id(value.at("id"), "added_tokens.id");
                if (!id) [[unlikely]] {
                    return std::unexpected(std::move(id).error());
                }
                auto token = parse_added_token(value, *id, "added_tokens");
                if (!token) [[unlikely]] {
                    return std::unexpected(std::move(token).error());
                }
                auto added = add_added_token(*impl, std::move(*token));
                if (!added) [[unlikely]] {
                    return std::unexpected(std::move(added).error());
                }
            }
        }

        const auto & config = *config_document;
        if (config.contains("added_tokens_decoder")) {
            const auto & added_tokens_decoder = config.at("added_tokens_decoder");
            if (!added_tokens_decoder.is_object()) [[unlikely]] {
                return std::unexpected(
                    invalid_configuration("added_tokens_decoder must be an object")
                );
            }

            for (const auto & [id_text, value] : added_tokens_decoder.items()) {
                auto id = parse_token_id_text(id_text, "added_tokens_decoder");
                if (!id) [[unlikely]] {
                    return std::unexpected(std::move(id).error());
                }
                auto token = parse_added_token(value, *id, "added_tokens_decoder");
                if (!token) [[unlikely]] {
                    return std::unexpected(std::move(token).error());
                }
                auto added = add_added_token(*impl, std::move(*token));
                if (!added) [[unlikely]] {
                    return std::unexpected(std::move(added).error());
                }
            }
        }

        std::sort(
            impl->added_tokens.begin(),
            impl->added_tokens.end(),
            [](const AddedToken & left, const AddedToken & right) {
                if (left.content.size() != right.content.size()) {
                    return left.content.size() > right.content.size();
                }
                return left.content < right.content;
            }
        );

        auto bos_token_id = read_special_token_id(config, "bos_token", *impl);
        if (!bos_token_id) [[unlikely]] {
            return std::unexpected(std::move(bos_token_id).error());
        }
        impl->bos_token_id = *bos_token_id;

        auto eos_token_id = read_special_token_id(config, "eos_token", *impl);
        if (!eos_token_id) [[unlikely]] {
            return std::unexpected(std::move(eos_token_id).error());
        }
        impl->eos_token_id = *eos_token_id;

        auto pad_token_id = read_special_token_id(config, "pad_token", *impl);
        if (!pad_token_id) [[unlikely]] {
            return std::unexpected(std::move(pad_token_id).error());
        }
        impl->pad_token_id = *pad_token_id;

        impl->byte_level = make_byte_level_codec();
        return Qwen3Tokenizer(std::move(impl));
    } catch (const nlohmann::json::exception &) {
        return std::unexpected(
            invalid_configuration("Qwen3 tokenizer contains an invalid JSON structure")
        );
    }
}

std::expected<std::vector<common::TokenId>, TokenizerError>
Qwen3Tokenizer::encode(std::string_view text, bool add_special_tokens) const
{
    static_cast<void>(add_special_tokens);

    std::vector<common::TokenId> output;
    if (text.empty()) {
        return output;
    }

    std::size_t ordinary_begin = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const AddedToken * match = nullptr;
        for (const auto & token : impl_->added_tokens) {
            if (token.content.size() <= text.size() - offset &&
                text.compare(offset, token.content.size(), token.content) == 0) {
                match = &token;
                break;
            }
        }

        if (match == nullptr) {
            ++offset;
            continue;
        }

        if (offset > ordinary_begin) {
            auto result = encode_plain_text(
                *impl_,
                text.substr(ordinary_begin, offset - ordinary_begin),
                output
            );
            if (!result) [[unlikely]] {
                return std::unexpected(std::move(result).error());
            }
        }
        output.push_back(match->id);
        offset += match->content.size();
        ordinary_begin = offset;
    }

    if (ordinary_begin < text.size()) {
        auto result = encode_plain_text(*impl_, text.substr(ordinary_begin), output);
        if (!result) [[unlikely]] {
            return std::unexpected(std::move(result).error());
        }
    }

    return output;
}

std::expected<std::string, TokenizerError>
Qwen3Tokenizer::decode(std::span<const common::TokenId> token_ids, bool skip_special_tokens) const
{
    std::string byte_buffer;
    std::string output;
    for (const auto token_id : token_ids) {
        auto result =
            append_decoded_token(*impl_, token_id, byte_buffer, output, skip_special_tokens);
        if (!result) [[unlikely]] {
            return std::unexpected(std::move(result).error());
        }
    }

    if (!byte_buffer.empty()) {
        output += decode_utf8_with_replacement(byte_buffer);
    }
    return output;
}

std::size_t Qwen3Tokenizer::vocab_size() const noexcept
{
    return impl_->base_vocab_size;
}

std::size_t Qwen3Tokenizer::total_vocab_size() const noexcept
{
    return impl_->id_to_token.size();
}

std::optional<common::TokenId> Qwen3Tokenizer::token_id(std::string_view token) const noexcept
{
    const auto found = impl_->token_to_id.find(std::string(token));
    if (found == impl_->token_to_id.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::optional<common::TokenId> Qwen3Tokenizer::bos_token_id() const noexcept
{
    return impl_->bos_token_id;
}

std::optional<common::TokenId> Qwen3Tokenizer::eos_token_id() const noexcept
{
    return impl_->eos_token_id;
}

std::optional<common::TokenId> Qwen3Tokenizer::pad_token_id() const noexcept
{
    return impl_->pad_token_id;
}

} // namespace liteinfer::core::tokenizer::qwen3
