// Model-independent unit tests for boosted TDT GREEDY token selection
// (pk::detail::boosted_greedy_token) — the second stage of NeMo's GPU-PB
// greedy decoding.
//
// These drive the selection rule directly with synthetic token logits, so they
// are deterministic and need no GGUF. The multi-step cases walk the phrase
// state exactly as tdt_greedy does: carry the returned state forward, and only
// on a non-blank emission.

#include "boosting_tree.hpp"
#include "search.hpp"
#include "tdt.hpp"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "test_tdt_greedy_boost: FAILED: %s\n",
                     what.c_str());
        ++failures;
    }
}

void check_eq(int got, int want, const std::string& what) {
    if (got != want) {
        std::fprintf(stderr,
                     "test_tdt_greedy_boost: FAILED: %s (got %d, want %d)\n",
                     what.c_str(), got, want);
        ++failures;
    }
}

// Vocabulary layout shared by these tests: ids 0..3 are real tokens, 4 = blank.
constexpr int kBlank = 4;

pk::BoostingConfig make_config(const pk::BoostingTree& tree, float alpha) {
    pk::BoostingConfig cfg;
    cfg.tree = &tree;
    cfg.alpha = alpha;
    return cfg;
}

pk::BoostingTree::Params nemo_params() {
    pk::BoostingTree::Params p;
    p.context_score = 1.0f;
    p.depth_scaling = 2.0f;
    return p;
}

// The unboosted reference: plain argmax over the token slice, which is what
// tdt_greedy does when boosting is inactive.
int plain_argmax(const std::vector<float>& logits) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i)
        if (logits[i] > logits[best]) best = (int)i;
    return best;
}

// ---------------------------------------------------------------------------

void test_boost_flips_an_ambiguous_choice() {
    // Phrase "1 2". Acoustically token 0 leads token 1 by a hair; a boost on
    // the phrase's first token must flip the selection to 1.
    const pk::BoostingTree tree({{1, 2}}, nemo_params());
    //                     tok0   tok1   tok2   tok3   blank
    const std::vector<float> lg{-1.0f, -1.2f, -5.0f, -5.0f, -3.0f};

    check_eq(plain_argmax(lg), 0, "unboosted argmax picks token 0");

    pk::BoostingTree::State next = pk::BoostingTree::kRoot;
    const int k = pk::detail::boosted_greedy_token(
        lg.data(), kBlank, make_config(tree, 1.0f),
        pk::BoostingTree::kRoot, &next);
    check_eq(k, 1, "boost flips the ambiguous choice to the key token");
    check(next != pk::BoostingTree::kRoot,
          "phrase state advanced past the root after matching token 1");
}

void test_alpha_zero_matches_unboosted() {
    // Requirement: alpha == 0 must reproduce the unboosted argmax exactly,
    // for every logit arrangement, including ties.
    const pk::BoostingTree tree({{1, 2}, {3}}, nemo_params());
    const std::vector<std::vector<float>> cases = {
        {-1.0f, -1.2f, -5.0f, -5.0f, -3.0f},
        {-9.0f, -9.0f, -9.0f, -9.0f, -0.1f},   // blank wins outright
        {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f},   // all tied -> lowest id
        {-4.0f, -3.0f, -2.0f, -1.0f, -9.0f},
        {-1.0f, -2.0f, -3.0f, -0.2f, -8.0f},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        pk::BoostingTree::State next = pk::BoostingTree::kRoot;
        const int boosted = pk::detail::boosted_greedy_token(
            cases[i].data(), kBlank, make_config(tree, 0.0f),
            pk::BoostingTree::kRoot, &next);
        check_eq(boosted, plain_argmax(cases[i]),
                 "alpha=0 matches unboosted argmax, case " + std::to_string(i));
    }
}

void test_unrelated_phrase_does_not_change_selection() {
    // A phrase over tokens the audio does not contain must leave every
    // selection untouched, at any alpha.
    const pk::BoostingTree tree({{3}}, nemo_params());
    const std::vector<std::vector<float>> cases = {
        {-1.0f, -1.2f, -5.0f, -9.0f, -3.0f},
        {-2.0f, -0.5f, -3.0f, -9.5f, -4.0f},
    };
    for (float alpha : {0.5f, 2.0f, 5.0f}) {
        for (size_t i = 0; i < cases.size(); ++i) {
            pk::BoostingTree::State next = pk::BoostingTree::kRoot;
            const int boosted = pk::detail::boosted_greedy_token(
                cases[i].data(), kBlank, make_config(tree, alpha),
                pk::BoostingTree::kRoot, &next);
            check_eq(boosted, plain_argmax(cases[i]),
                     "unrelated phrase leaves selection unchanged (alpha " +
                         std::to_string(alpha) + ", case " +
                         std::to_string(i) + ")");
        }
    }
}

void test_blank_is_never_boosted() {
    // When blank dominates every token by more than any boost can bridge, the
    // decoder must still emit blank: biasing changes WHICH token is emitted,
    // never WHETHER one is.
    const pk::BoostingTree tree({{1, 2}}, nemo_params());
    const std::vector<float> lg{-40.0f, -40.0f, -40.0f, -40.0f, -0.01f};

    pk::BoostingTree::State next = pk::BoostingTree::kRoot;
    const int k = pk::detail::boosted_greedy_token(
        lg.data(), kBlank, make_config(tree, 2.0f),
        pk::BoostingTree::kRoot, &next);
    check_eq(k, kBlank, "blank still wins against a strong boost");
    check(next == pk::BoostingTree::kRoot,
          "a blank win leaves the phrase state untouched");
}

void test_blank_does_not_advance_or_reset_phrase_state() {
    // Requirement 4/5: a blank step mid-phrase must neither advance nor reset
    // the match. Enter the phrase, take a blank step, then finish the phrase.
    const pk::BoostingTree tree({{1, 2}}, nemo_params());
    const pk::BoostingConfig cfg = make_config(tree, 1.0f);

    // Step 1: match token 1.
    const std::vector<float> lg1{-5.0f, -1.0f, -5.0f, -5.0f, -6.0f};
    pk::BoostingTree::State s = pk::BoostingTree::kRoot;
    pk::BoostingTree::State after1 = s;
    check_eq(pk::detail::boosted_greedy_token(lg1.data(), kBlank, cfg, s,
                                              &after1),
             1, "step 1 emits the phrase's first token");
    check(after1 != pk::BoostingTree::kRoot, "step 1 advanced the phrase");

    // Step 2: blank dominates. State must survive unchanged.
    const std::vector<float> lg2{-40.0f, -40.0f, -40.0f, -40.0f, -0.01f};
    pk::BoostingTree::State after2 = after1;
    check_eq(pk::detail::boosted_greedy_token(lg2.data(), kBlank, cfg, after1,
                                              &after2),
             kBlank, "step 2 emits blank");
    check(after2 == after1, "blank preserved the mid-phrase state exactly");

    // Step 3: token 2 completes the phrase despite trailing token 0 slightly.
    const std::vector<float> lg3{-1.0f, -5.0f, -1.3f, -5.0f, -6.0f};
    check_eq(plain_argmax(lg3), 0, "unboosted step 3 would pick token 0");
    pk::BoostingTree::State after3 = after2;
    check_eq(pk::detail::boosted_greedy_token(lg3.data(), kBlank, cfg, after2,
                                              &after3),
             2, "step 3 completes the phrase across the blank");
}

void test_phrase_progress_mismatch_and_reset() {
    // Depth rewards grow, so the deeper a match runs the harder it holds. On a
    // mismatch the state must fall back to the root (no phrase continues).
    const pk::BoostingTree tree({{1, 2, 3}}, nemo_params());
    const pk::BoostingConfig cfg = make_config(tree, 1.0f);

    pk::BoostingTree::State s = pk::BoostingTree::kRoot;
    const std::vector<float> step1{-5.0f, -1.0f, -5.0f, -5.0f, -6.0f};
    pk::detail::boosted_greedy_token(step1.data(), kBlank, cfg, s, &s);
    const pk::BoostingTree::State depth1 = s;
    check(depth1 != pk::BoostingTree::kRoot, "advanced to depth 1");

    const std::vector<float> step2{-5.0f, -5.0f, -1.0f, -5.0f, -6.0f};
    pk::detail::boosted_greedy_token(step2.data(), kBlank, cfg, s, &s);
    check(s != depth1 && s != pk::BoostingTree::kRoot, "advanced to depth 2");

    // Token 0 continues nothing: the state must reset to the root.
    const std::vector<float> mismatch{-0.1f, -9.0f, -9.0f, -9.0f, -9.0f};
    pk::BoostingTree::State after = s;
    check_eq(pk::detail::boosted_greedy_token(mismatch.data(), kBlank, cfg, s,
                                              &after),
             0, "mismatching token still wins on acoustics");
    check(after == pk::BoostingTree::kRoot,
          "a mismatch resets the phrase state to the root");
}

void test_completion_and_repeated_occurrences() {
    // The same phrase must boost every time it occurs, not just the first.
    const pk::BoostingTree tree({{1, 2}}, nemo_params());
    const pk::BoostingConfig cfg = make_config(tree, 1.0f);

    // Token 1 leads slightly; token 2 trails token 0 so only a live phrase
    // match can pull it ahead.
    const std::vector<float> first{-5.0f, -1.0f, -5.0f, -5.0f, -6.0f};
    const std::vector<float> second{-1.0f, -5.0f, -1.3f, -5.0f, -6.0f};

    pk::BoostingTree::State s = pk::BoostingTree::kRoot;
    for (int occurrence = 0; occurrence < 3; ++occurrence) {
        check_eq(pk::detail::boosted_greedy_token(first.data(), kBlank, cfg, s,
                                                  &s),
                 1, "occurrence " + std::to_string(occurrence) + ": first token");
        check_eq(pk::detail::boosted_greedy_token(second.data(), kBlank, cfg, s,
                                                  &s),
                 2, "occurrence " + std::to_string(occurrence) +
                        ": phrase completed");
    }
}

void test_state_does_not_leak_between_runs() {
    // Requirement 11: state is a caller-owned value, so two runs starting at
    // kRoot cannot influence each other even after a deep partial match.
    const pk::BoostingTree tree({{1, 2}}, nemo_params());
    const pk::BoostingConfig cfg = make_config(tree, 1.0f);
    const std::vector<float> enter{-5.0f, -1.0f, -5.0f, -5.0f, -6.0f};
    const std::vector<float> probe{-1.0f, -5.0f, -1.3f, -5.0f, -6.0f};

    // Run A leaves the state mid-phrase.
    pk::BoostingTree::State a = pk::BoostingTree::kRoot;
    pk::detail::boosted_greedy_token(enter.data(), kBlank, cfg, a, &a);
    check(a != pk::BoostingTree::kRoot, "run A is mid-phrase");

    // Run B starts fresh: `probe` must fall back to the acoustic winner,
    // proving run A's progress did not carry over.
    pk::BoostingTree::State b = pk::BoostingTree::kRoot;
    check_eq(pk::detail::boosted_greedy_token(probe.data(), kBlank, cfg,
                                              pk::BoostingTree::kRoot, &b),
             0, "a fresh run is unaffected by the previous run's state");
}

// An inactive config must leave decoding bit-identical. Covers the two ways a
// caller disables boosting (alpha == 0, empty tree) over randomized logits with
// many deliberate ties — the case where a wrong tie-break would show up.
void test_ctc_unchanged_when_inactive() {
    std::mt19937 rng(99);
    const int V = 8, blank = 7;   // vocab_plus_1 = 8, blank is the last class
    const pk::BoostingTree tree({{1, 2}, {3}}, nemo_params());
    const pk::BoostingTree empty_tree;

    int mismatches = 0;
    for (int trial = 0; trial < 500; ++trial) {
        const int T = 1 + (int)(rng() % 12);
        std::vector<float> logits((size_t)T * V);
        for (float& x : logits) x = -(float)(rng() % 5);   // force many ties

        const std::vector<int32_t> baseline =
            pk::ctc_greedy(logits, T, V, blank);

        if (pk::ctc_greedy(logits, T, V, blank, nullptr,
                           make_config(tree, 0.0f)) != baseline)
            ++mismatches;
        if (pk::ctc_greedy(logits, T, V, blank, nullptr,
                           make_config(empty_tree, 3.0f)) != baseline)
            ++mismatches;
    }
    check_eq(mismatches, 0,
             "ctc_greedy is bit-identical with alpha=0 and with an empty tree");
}

}  // namespace

int main() {
    test_boost_flips_an_ambiguous_choice();
    test_alpha_zero_matches_unboosted();
    test_unrelated_phrase_does_not_change_selection();
    test_blank_is_never_boosted();
    test_blank_does_not_advance_or_reset_phrase_state();
    test_phrase_progress_mismatch_and_reset();
    test_completion_and_repeated_occurrences();
    test_state_does_not_leak_between_runs();
    test_ctc_unchanged_when_inactive();

    if (failures != 0) {
        std::fprintf(stderr, "test_tdt_greedy_boost: %d check(s) failed\n",
                     failures);
        return 1;
    }
    return 0;
}
