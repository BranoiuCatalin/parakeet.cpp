#include "tokenizer.hpp"
#include <algorithm>
namespace pk {
// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece meta-space marker.
// UTF-8 encoding: 0xE2 0x96 0x81  (3 bytes)
static const char META_SPACE[] = "\xe2\x96\x81";
static const size_t META_SPACE_LEN = 3;

std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids) {
    // Step 1: concatenate the piece strings for each id.
    std::string result;
    result.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < pieces.size()) {
            result += pieces[(size_t)id];
        }
    }

    // Step 2: replace every occurrence of META_SPACE (▁) with a regular space.
    std::string out;
    out.reserve(result.size());
    for (size_t i = 0; i < result.size(); ) {
        // Check for the 3-byte UTF-8 sequence 0xE2 0x96 0x81
        if (i + META_SPACE_LEN <= result.size() &&
            (unsigned char)result[i]   == 0xE2 &&
            (unsigned char)result[i+1] == 0x96 &&
            (unsigned char)result[i+2] == 0x81) {
            out += ' ';
            i += META_SPACE_LEN;
        } else {
            out += result[i++];
        }
    }

    // Step 3: strip a single leading space (SentencePiece decode_ids behavior).
    if (!out.empty() && out[0] == ' ') {
        out.erase(0, 1);
    }
    return out;
}

namespace {
// Normalize a phrase to SentencePiece surface form: trim, collapse internal
// whitespace, and write each space as the ▁ meta-space marker (including a
// leading one, since a key phrase always begins a word).
std::string to_metaspace(const std::string& phrase) {
    std::string out;
    out.reserve(phrase.size() + META_SPACE_LEN);

    size_t i = 0;
    const auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
               c == '\f' || c == '\v';
    };
    while (i < phrase.size() && is_space((unsigned char)phrase[i])) ++i;

    bool pending_space = true;   // leading word start
    for (; i < phrase.size(); ++i) {
        if (is_space((unsigned char)phrase[i])) {
            pending_space = true;
            continue;
        }
        if (pending_space) {
            out += META_SPACE;
            pending_space = false;
        }
        out += phrase[i];
    }
    return out;
}
}  // namespace

std::vector<int32_t> encode_phrase_tokens(const std::vector<std::string>& pieces,
                                          const std::string& phrase) {
    const std::string text = to_metaspace(phrase);
    if (text.empty()) return {};

    // Longest piece length bounds the match window, so a huge vocabulary does
    // not turn every position into a full scan of the remaining text.
    size_t max_piece = 0;
    for (const std::string& piece : pieces)
        if (piece.size() > max_piece) max_piece = piece.size();
    if (max_piece == 0) return {};

    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        int32_t best_id = -1;
        size_t best_len = 0;

        const size_t limit = std::min(max_piece, text.size() - pos);
        for (size_t id = 0; id < pieces.size(); ++id) {
            const std::string& piece = pieces[id];
            if (piece.empty() || piece.size() > limit ||
                piece.size() <= best_len) {
                continue;
            }
            // Skip bracketed special tokens (<unk>, <EOU>, language tags): a
            // literal "<en-US>" in a phrase must not resolve to that control id.
            if ((piece.front() == '<' && piece.back() == '>') ||
                (piece.front() == '[' && piece.back() == ']')) {
                continue;
            }
            if (text.compare(pos, piece.size(), piece) == 0) {
                best_id = (int32_t)id;
                best_len = piece.size();
            }
        }

        if (best_id < 0) return {};   // uncoverable phrase
        ids.push_back(best_id);
        pos += best_len;
    }
    return ids;
}

std::vector<int32_t> strip_special_tokens(const std::vector<std::string>& pieces,
                                          const std::vector<int32_t>& ids) {
    std::vector<int32_t> out;
    out.reserve(ids.size());
    for (int32_t id : ids) {
        if (id >= 0 && (size_t)id < pieces.size()) {
            const std::string& piece = pieces[(size_t)id];
            if (!piece.empty() &&
                ((piece.front() == '<' && piece.back() == '>') ||
                 (piece.front() == '[' && piece.back() == ']'))) {
                continue;
            }
        }
        out.push_back(id);
    }
    return out;
}
} // namespace pk
