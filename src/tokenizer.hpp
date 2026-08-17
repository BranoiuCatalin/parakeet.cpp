#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace pk {
// Map each id to its SentencePiece piece, concatenate, replace the U+2581
// meta-space character (▁, UTF-8: 0xE2 0x96 0x81) with a regular space, and
// strip a single leading space if present.  This matches the behavior of
// NeMo SentencePieceTokenizer::ids_to_text (non-legacy path) which calls
// sentencepiece::SentencePieceProcessor::decode_ids.
std::string detokenize(const std::vector<std::string>& pieces,
                       const std::vector<int32_t>& ids);

// Drop ids whose piece is a bracketed special token (`<...>` or `[...]`, e.g.
// language tags like `<en-US>` or `<EOU>`), so they never reach detokenize().
// Ordinary SentencePiece content tokens (including `▁`-prefixed word starts)
// are left untouched.
std::vector<int32_t> strip_special_tokens(const std::vector<std::string>& pieces,
                                          const std::vector<int32_t>& ids);

// Encode a context-biasing key phrase to token ids against the model's own
// SentencePiece vocabulary — the inverse direction of detokenize(), needed to
// build a pk::BoostingTree from user-supplied phrases.
//
// This is a greedy longest-match (maximum munch) segmentation, NOT the Viterbi
// unigram segmentation SentencePiece itself would run: the GGUF stores only the
// piece strings, not their unigram log-probabilities, so the true scoring
// segmentation is not recoverable. For boosting this is the right trade-off —
// the tree only needs A tokenization the decoder can plausibly emit, and greedy
// longest-match reproduces SentencePiece's output on the overwhelming majority
// of ordinary words. Callers that need exactness can pass token ids directly.
//
// The input is normalized the way SentencePiece encodes text: leading/trailing
// whitespace trimmed, internal whitespace runs collapsed to one space, and each
// space rendered as the U+2581 meta-space marker (word starts are `▁`-prefixed).
// Matching is byte-exact and case-sensitive — the caller decides casing, since
// these models emit lowercase for English but cased text for some locales.
//
// Returns an empty vector when the phrase is empty or when any part of it
// cannot be covered by the vocabulary (e.g. it contains a character the model
// has no piece for). An empty result means "unusable phrase" and should be
// reported to the user rather than silently boosted as nothing.
std::vector<int32_t> encode_phrase_tokens(const std::vector<std::string>& pieces,
                                          const std::string& phrase);
} // namespace pk
