// Model-independent unit tests for the context-biasing phrase tree
// (pk::BoostingTree) and the phrase tokenizer (pk::encode_phrase_tokens).

#include "boosting_tree.hpp"
#include "tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "test_boosting_tree: FAILED: %s\n", what);
        ++failures;
    }
}

void check_near(float got, float want, const char* what) {
    if (std::fabs(got - want) > 1e-4f) {
        std::fprintf(stderr,
                     "test_boosting_tree: FAILED: %s (got %.6f, want %.6f)\n",
                     what, got, want);
        ++failures;
    }
}

// Total boost accumulated by walking `tokens` from the root, including the
// final-state refund — i.e. exactly what a decoder would add to a hypothesis.
float total_boost(const pk::BoostingTree& tree,
                  const std::vector<int32_t>& tokens) {
    pk::BoostingTree::State state = pk::BoostingTree::kRoot;
    float sum = 0.0f;
    for (int32_t token : tokens) {
        pk::BoostingTree::State next = pk::BoostingTree::kRoot;
        sum += tree.advance(state, token, &next);
        state = next;
    }
    return sum + tree.final_score(state);
}

void test_empty_tree_is_inert() {
    const pk::BoostingTree tree;
    check(tree.empty(), "default-constructed tree is empty");

    pk::BoostingTree::State next = 99;
    check_near(tree.advance(pk::BoostingTree::kRoot, 7, &next), 0.0f,
               "empty tree gives no boost");
    check(next == pk::BoostingTree::kRoot, "empty tree resets to root");

    pk::BoostingConfig cfg;
    cfg.tree = &tree;
    cfg.alpha = 3.0f;
    check(!cfg.active(), "config with an empty tree is inactive");

    const pk::BoostingTree nonempty({{1, 2}}, {});
    pk::BoostingConfig zero_alpha;
    zero_alpha.tree = &nonempty;
    zero_alpha.alpha = 0.0f;
    check(!zero_alpha.active(), "config with alpha 0 is inactive");
}

void test_arc_scores_follow_nemo_formula() {
    pk::BoostingTree::Params params;
    params.context_score = 1.0f;
    params.depth_scaling = 2.0f;
    const pk::BoostingTree tree({{10, 11, 12}}, params);

    check(!tree.empty(), "tree with a phrase is not empty");
    check(tree.phrase_count() == 1, "one phrase counted");

    // depth 1 -> context_score; depth d>1 -> context_score*depth_scaling + ln(d)
    pk::BoostingTree::State s = pk::BoostingTree::kRoot;
    check_near(tree.advance(s, 10, &s), 1.0f, "depth-1 arc score");
    check_near(tree.advance(s, 11, &s), 2.0f + std::log(2.0f),
               "depth-2 arc score");
    check_near(tree.advance(s, 12, &s), 2.0f + std::log(3.0f),
               "depth-3 arc score");
    check_near(tree.final_score(s), 0.0f, "completed phrase owes no refund");
}

void test_partial_match_is_fully_refunded() {
    // The property that keeps biasing from hallucinating: a hypothesis that
    // starts a phrase and abandons it must end up with exactly zero net boost.
    pk::BoostingTree::Params params;
    params.context_score = 1.0f;
    params.depth_scaling = 2.0f;
    const pk::BoostingTree tree({{10, 11, 12}}, params);

    check_near(total_boost(tree, {10}), 0.0f,
               "abandoned 1-token prefix nets zero");
    check_near(total_boost(tree, {10, 11}), 0.0f,
               "abandoned 2-token prefix nets zero");
    check_near(total_boost(tree, {10, 11, 99}), 0.0f,
               "prefix broken by a foreign token nets zero");
    check_near(total_boost(tree, {99, 98}), 0.0f,
               "unrelated tokens net zero");

    // A completed phrase keeps the full accumulated reward.
    const float want = 1.0f + (2.0f + std::log(2.0f)) + (2.0f + std::log(3.0f));
    check_near(total_boost(tree, {10, 11, 12}), want,
               "completed phrase keeps its reward");
    check_near(total_boost(tree, {10, 11, 12, 99}), want,
               "reward survives tokens after the phrase");
}

void test_failure_links_resume_mid_phrase() {
    // "ab" must still be found when the decoder emitted "a a b": after the
    // second 'a' breaks the first match, the failure link lands on the node for
    // 'a' rather than the root, so the following 'b' completes the phrase.
    pk::BoostingTree::Params params;
    params.context_score = 1.0f;
    params.depth_scaling = 2.0f;
    const pk::BoostingTree tree({{1, 2}}, params);

    const float want = 1.0f + (2.0f + std::log(2.0f));
    check_near(total_boost(tree, {1, 1, 2}), want,
               "failure link recovers the match after a repeat");
    check_near(total_boost(tree, {5, 1, 2}), want,
               "phrase found after an unrelated prefix");
}

void test_shared_prefix_and_duplicates() {
    pk::BoostingTree::Params params;
    params.context_score = 1.0f;
    params.depth_scaling = 2.0f;
    const pk::BoostingTree tree({{1, 2}, {1, 3}, {1, 2}}, params);

    // The duplicate collapses; the shared prefix is stored once.
    check(tree.phrase_count() == 2, "duplicate phrase counted once");

    const float want_12 = 1.0f + (2.0f + std::log(2.0f));
    check_near(total_boost(tree, {1, 2}), want_12, "first branch scored");
    check_near(total_boost(tree, {1, 3}), want_12, "second branch scored");
}

void test_unk_score_applies_off_phrase() {
    pk::BoostingTree::Params params;
    params.context_score = 1.0f;
    params.depth_scaling = 2.0f;
    params.unk_score = -0.5f;
    const pk::BoostingTree tree({{1, 2}}, params);

    pk::BoostingTree::State s = pk::BoostingTree::kRoot;
    check_near(tree.advance(s, 42, &s), -0.5f, "off-phrase token gets unk_score");
    check(s == pk::BoostingTree::kRoot, "off-phrase token stays at the root");
}

void test_encode_phrase_tokens() {
    // A miniature SentencePiece-style vocabulary. MS is U+2581 (▁), written as
    // a separate literal so the following letter is not swallowed into the hex
    // escape ("\xe2\x96\x81a" would parse as one out-of-range escape).
    const std::string MS = "\xe2\x96\x81";
    const std::vector<std::string> pieces = {
        "<unk>",       // 0  special: must never be matched literally
        MS,            // 1  bare metaspace
        MS + "a",      // 2
        MS + "ab",     // 3
        "b",           // 4
        "c",           // 5
        MS + "cd",     // 6
    };

    // Longest match wins: "▁ab" (3) beats "▁a"(2) + "b"(4).
    const std::vector<int32_t> ab = pk::encode_phrase_tokens(pieces, "ab");
    check(ab == std::vector<int32_t>{3}, "longest-match segmentation");

    // Word starts get the metaspace prefix; internal words too.
    const std::vector<int32_t> ab_cd =
        pk::encode_phrase_tokens(pieces, "ab cd");
    check(ab_cd == (std::vector<int32_t>{3, 6}), "two words encode");

    // Whitespace is normalized (leading/trailing trimmed, runs collapsed).
    check(pk::encode_phrase_tokens(pieces, "  ab\t cd  ") == ab_cd,
          "whitespace normalized");

    // Fallback to shorter pieces when the long one does not fit.
    const std::vector<int32_t> ac = pk::encode_phrase_tokens(pieces, "ac");
    check(ac == (std::vector<int32_t>{2, 5}), "falls back to shorter pieces");

    // Uncoverable phrases report failure rather than boosting nothing.
    check(pk::encode_phrase_tokens(pieces, "zz").empty(),
          "uncoverable phrase returns empty");
    check(pk::encode_phrase_tokens(pieces, "").empty(),
          "empty phrase returns empty");
    check(pk::encode_phrase_tokens(pieces, "   ").empty(),
          "whitespace-only phrase returns empty");

    // A literal "<unk>" must not resolve to the special token id 0.
    check(pk::encode_phrase_tokens(pieces, "<unk>").empty(),
          "special tokens are not matched literally");
}

}  // namespace

int main() {
    test_empty_tree_is_inert();
    test_arc_scores_follow_nemo_formula();
    test_partial_match_is_fully_refunded();
    test_failure_links_resume_mid_phrase();
    test_shared_prefix_and_duplicates();
    test_unk_score_applies_off_phrase();
    test_encode_phrase_tokens();

    if (failures != 0) {
        std::fprintf(stderr, "test_boosting_tree: %d check(s) failed\n",
                     failures);
        return 1;
    }
    return 0;
}
