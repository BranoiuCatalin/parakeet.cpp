#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pk {

// Context-biasing ("word boosting") phrase tree — a CPU port of NeMo's
// GPU-PB / TurboBias phrase-boosting tree (arXiv 2508.07014), the mechanism
// behind `boosting_tree` in NeMo's CTC/RNNT/TDT decoding configs.
//
// The tree is an Aho-Corasick automaton over the TOKENIZED key phrases. During
// decoding each hypothesis carries one tree state; every candidate non-blank
// token is scored by advance(), and the returned boost is added to the acoustic
// log-probability in shallow fusion:
//
//     score = acoustic_logp + alpha * boost
//
// with `alpha` = NeMo's `boosting_tree_alpha`. No retraining is involved; this
// is purely a decode-time rescoring of tokens the model already knows.
//
// Arc scores (NeMo build_gpu_boosting_tree defaults, see BoostingParams):
//
//     arc_score(depth) = context_score                            if depth == 1
//                      = context_score * depth_scaling + ln(depth) if depth > 1
//
// The reward therefore grows as a phrase is matched further, which keeps the
// decoder committed to a partially-matched phrase instead of abandoning it at
// the first acoustically weak token.
//
// Backoff / score cancellation is what makes this safe. Each node stores the
// accumulated score along its path from the root. When a match breaks, the
// automaton follows failure links to the longest proper suffix that is still a
// valid prefix, and the score difference
//
//     backoff = acc_score(fail_state) - acc_score(current_state)
//
// is applied — REFUNDING the boost already handed out for the partial match
// that did not pan out. Without this cancellation a partially-matched phrase
// keeps its unearned reward and the decoder hallucinates key phrases; with it,
// a hypothesis that never completes a phrase ends up exactly where it started.
// Final (phrase-completing) nodes keep a zero backoff so a completed phrase is
// never refunded.
//
// The tree is model-independent and holds no ggml/GGUF state: it is built from
// token-id sequences (see pk::encode_phrase_tokens) and unit-tested standalone.
class BoostingTree {
public:
    // Tuning knobs. Names and defaults mirror NeMo's
    // scripts/asr_context_biasing/build_gpu_boosting_tree.py so values tuned
    // against NeMo transfer directly.
    struct Params {
        // Score for each arc transition (NeMo `context_score`, recommended 1.0).
        float context_score = 1.0f;
        // Depth scaling factor (NeMo `depth_scaling`). 2.0 is NeMo's
        // recommendation for CTC / RNN-T / TDT (1.0 is for Canary/AED).
        float depth_scaling = 2.0f;
        // Score for the root self-loop on non-phrase-initial tokens
        // (NeMo `unk_score`). 0.0 leaves non-key tokens untouched.
        float unk_score = 0.0f;
    };

    // A hypothesis's position in the automaton. Cheap to copy (one int), so
    // every beam hypothesis can carry its own without allocation.
    using State = int32_t;

    static constexpr State kRoot = 0;

    BoostingTree() = default;

    // Build from tokenized phrases. Empty phrases are ignored. Duplicate
    // phrases collapse onto the same path (scored once, not twice).
    BoostingTree(const std::vector<std::vector<int32_t>>& phrases,
                 const Params& params);

    // True when no phrase was added — decoders use this to skip boosting
    // entirely and keep the unboosted path bit-identical.
    bool empty() const { return nodes_.size() <= 1; }

    size_t phrase_count() const { return phrase_count_; }
    size_t node_count() const { return nodes_.size(); }
    const Params& params() const { return params_; }

    // Score `token` from `state`.
    //
    // Returns the boost to add (scaled by alpha at the call site) and writes
    // the successor state to *next. Following the failure chain, the returned
    // value already nets out the backoff refund for the abandoned prefix, so
    // callers just add it — no separate cancellation bookkeeping.
    //
    // A token that continues no phrase from any suffix of the current path
    // returns the accumulated refund plus `unk_score` and lands back at kRoot.
    float advance(State state, int32_t token, State* next) const;

    // Refund still owed by a hypothesis that ENDS mid-phrase. Add this once at
    // the end of decoding so an incomplete trailing match keeps no reward.
    // Zero at the root and at any phrase-final node.
    float final_score(State state) const;

private:
    struct Node {
        // Outgoing arcs by token id -> child node index.
        std::unordered_map<int32_t, int32_t> next;
        // Accumulated arc score from the root along this path.
        float acc_score = 0.0f;
        // Aho-Corasick failure link (longest proper suffix that is a prefix).
        int32_t fail = kRoot;
        // True when a phrase ends here: its reward is earned and never refunded.
        bool is_final = false;
    };

    // Arc score at `depth` (1-based), per the formula in the class comment.
    float arc_score(int depth) const;

    // Resolve `token` from `state` following failure links; returns the child
    // node index or -1 when no suffix continues with `token`.
    int32_t walk(State state, int32_t token) const;

    std::vector<Node> nodes_{Node{}};   // nodes_[kRoot] is the root
    Params params_{};
    size_t phrase_count_ = 0;
};

// Context-biasing configuration threaded through the decoders.
//
// `tree` is borrowed, not owned, and must outlive the decode call. A null tree
// (or an empty one, or alpha == 0) disables boosting completely: the decoders
// take their original code path and produce bit-identical output, so nothing
// is paid for the feature when it is not used.
struct BoostingConfig {
    const BoostingTree* tree = nullptr;
    // Shallow-fusion weight (NeMo `boosting_tree_alpha`). Tune per dataset;
    // NeMo's examples typically land in the 1-4 range for CTC/TDT.
    float alpha = 0.0f;

    bool active() const {
        return tree != nullptr && !tree->empty() && alpha != 0.0f;
    }
};

} // namespace pk
