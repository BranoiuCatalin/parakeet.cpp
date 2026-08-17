#include "boosting_tree.hpp"

#include <cmath>
#include <deque>

namespace pk {

float BoostingTree::arc_score(int depth) const {
    if (depth <= 1) return params_.context_score;
    return params_.context_score * params_.depth_scaling +
           (float)std::log((double)depth);
}

BoostingTree::BoostingTree(const std::vector<std::vector<int32_t>>& phrases,
                           const Params& params)
    : params_(params) {
    // ---- Stage 1: prefix tree. Arc scores accumulate down each path. ----
    for (const std::vector<int32_t>& phrase : phrases) {
        if (phrase.empty()) continue;

        int32_t node = kRoot;
        int depth = 0;
        for (int32_t token : phrase) {
            ++depth;
            auto it = nodes_[(size_t)node].next.find(token);
            if (it != nodes_[(size_t)node].next.end()) {
                node = it->second;
                continue;
            }
            const float acc =
                nodes_[(size_t)node].acc_score + arc_score(depth);
            const int32_t child = (int32_t)nodes_.size();
            nodes_.push_back(Node{});
            nodes_.back().acc_score = acc;
            // Re-index the parent: push_back may have reallocated.
            nodes_[(size_t)node].next.emplace(token, child);
            node = child;
        }
        if (!nodes_[(size_t)node].is_final) {
            nodes_[(size_t)node].is_final = true;
            ++phrase_count_;
        }
    }

    // ---- Stage 2: Aho-Corasick failure links, breadth-first. ----
    // A node's failure target is the longest proper suffix of its path that is
    // also a prefix of some phrase, so a broken match resumes mid-phrase rather
    // than restarting at the root ("bay area" still matches inside "at bay area").
    std::deque<int32_t> queue;
    for (const auto& arc : nodes_[kRoot].next) {
        nodes_[(size_t)arc.second].fail = kRoot;
        queue.push_back(arc.second);
    }
    while (!queue.empty()) {
        const int32_t node = queue.front();
        queue.pop_front();

        // Copy the arc list: the loop below reads other nodes but the map we
        // iterate is stable, so a reference would be fine — the copy keeps the
        // intent (no mutation of `node` here) obvious and costs a small vector.
        std::vector<std::pair<int32_t, int32_t>> arcs(
            nodes_[(size_t)node].next.begin(), nodes_[(size_t)node].next.end());
        for (const auto& arc : arcs) {
            const int32_t token = arc.first;
            const int32_t child = arc.second;

            int32_t fail = nodes_[(size_t)node].fail;
            for (;;) {
                auto it = nodes_[(size_t)fail].next.find(token);
                if (it != nodes_[(size_t)fail].next.end() &&
                    it->second != child) {
                    fail = it->second;
                    break;
                }
                if (fail == kRoot) break;
                fail = nodes_[(size_t)fail].fail;
            }
            nodes_[(size_t)child].fail = fail;

            // A node whose suffix completes a phrase is itself terminal for
            // refund purposes: the shorter phrase's reward is earned.
            if (nodes_[(size_t)fail].is_final)
                nodes_[(size_t)child].is_final = true;

            queue.push_back(child);
        }
    }
}

int32_t BoostingTree::walk(State state, int32_t token) const {
    if (state < 0 || (size_t)state >= nodes_.size()) return -1;

    int32_t node = state;
    for (;;) {
        auto it = nodes_[(size_t)node].next.find(token);
        if (it != nodes_[(size_t)node].next.end()) return it->second;
        if (node == kRoot) return -1;
        node = nodes_[(size_t)node].fail;
    }
}

float BoostingTree::advance(State state, int32_t token, State* next) const {
    if (nodes_.size() <= 1 || state < 0 || (size_t)state >= nodes_.size()) {
        if (next) *next = kRoot;
        return 0.0f;
    }

    // A completed phrase has EARNED its reward: once past a final node there is
    // nothing left to refund, so the outstanding balance is zero rather than
    // the accumulated path score. Without this, the first token after a matched
    // phrase would claw the whole reward back.
    const float owed =
        nodes_[(size_t)state].is_final ? 0.0f : nodes_[(size_t)state].acc_score;
    const int32_t child = walk(state, token);

    if (child < 0) {
        // No suffix continues: refund whatever is still owed and drop back to
        // the root. `owed` is 0 at the root, so a non-key token in a
        // non-matching context correctly scores unk_score alone.
        if (next) *next = kRoot;
        return params_.unk_score - owed;
    }

    // Net delta between where we were and where we land. When `child` is a
    // direct arc this is exactly that arc's score; when we got there via a
    // failure link it also refunds the abandoned prefix.
    if (next) *next = child;
    return nodes_[(size_t)child].acc_score - owed;
}

float BoostingTree::final_score(State state) const {
    if (nodes_.size() <= 1 || state < 0 || (size_t)state >= nodes_.size())
        return 0.0f;
    if (nodes_[(size_t)state].is_final) return 0.0f;
    return -nodes_[(size_t)state].acc_score;
}

} // namespace pk
