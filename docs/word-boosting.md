# Word boosting (context biasing)

`parakeet.cpp` can bias decoding toward a caller-supplied list of key phrases —
names, jargon, product terms, spellings the model was never trained on. It is a
decode-time rescoring only: no retraining, no model surgery, and no change to
the GGUF.

Setting no phrases leaves every decoder on its original code path, so the
unboosted transcript is bit-identical to before this feature existed.

## Algorithm

This is a CPU port of NVIDIA NeMo's GPU-PB phrase-boosting tree, described in
[TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated
Phrase-Boosting Tree](https://arxiv.org/abs/2508.07014) and shipped in NeMo as
`boosting_tree` in the CTC/RNN-T/TDT decoding configs.

Key phrases are tokenized with the model's own SentencePiece vocabulary and
compiled into an Aho-Corasick prefix tree (`pk::BoostingTree`). Every hypothesis
carries one tree state. At each decoding step the candidate non-blank tokens are
rescored by shallow fusion:

```
score = acoustic_log_prob + alpha * boost
```

Arc scores follow NeMo's formula, so values tuned against NeMo transfer here:

```
arc_score(depth) = context_score                             if depth == 1
                 = context_score * depth_scaling + ln(depth)  if depth > 1
```

The reward grows with match depth, which keeps the decoder committed to a
partially matched phrase through an acoustically weak token in the middle of it.

### Score cancellation

The property that makes biasing safe rather than a hallucination generator is
the refund. Each node stores its accumulated score from the root; when a match
breaks, the automaton follows failure links and applies

```
backoff = acc_score(fail_state) - acc_score(current_state)
```

which gives back the reward already handed out for the prefix that did not pan
out. `final_score()` settles the same debt for a hypothesis that ends mid-phrase.
Phrase-final nodes are exempt: a completed phrase has earned its reward and
keeps it.

The net effect, verified by a randomized test over 3000 generated trees: **a
token sequence that never completes a phrase accumulates exactly zero boost, at
any alpha.** Biasing can only ever change which of several candidates wins, never
inflate a hypothesis that contains no key phrase at all.

Failure links also mean a phrase is still found when it does not start cleanly:
`ab` is matched inside `a a b`, because the broken match backs off to the node
for `a` rather than all the way to the root.

### Per-decoder application

| Decoder | How | Notes |
|---|---|---|
| `tdt_beam_search` | Boost the whole token slice before `top_k` | Strongest path. Promoting a key token INTO the beam is the point — an acoustically weak first token can still be recovered once later tokens confirm the phrase. |
| `tdt_greedy` | NeMo's two-stage greedy: argmax, rescore non-blanks, re-pick | Commits per frame, so a phrase whose first token scores very low is still missable. |
| `ctc_greedy` | Same two-stage rescoring per frame | Tree state advances only on an EMITTED token, so CTC's repeat-collapse does not advance a phrase five times for one token held across five frames. |

Two invariants hold across all three:

- **Blank is never boosted.** Non-blank candidates are compared against the
  unboosted blank score, so biasing can change *which* token is emitted but never
  *whether* one is.
- **Durations are never boosted.** Only the token slice is rescored, so TDT
  timing/timestamps cannot be distorted by biasing.

Reported confidences also stay honest: the CTC path keeps the acoustic
log-probability of the selected class for `TokenInfo.conf`, not the boosted
score (which could exceed a probability).

## Tuning

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` (`boosting_tree_alpha`) | 0 (off) | Shallow-fusion weight. The knob to tune. |
| `context_score` | 1.0 | Per-arc reward. NeMo recommends leaving at 1.0. |
| `depth_scaling` | 2.0 | Depth scaling. NeMo's value for CTC/RNN-T/TDT. |

Start at `alpha` 1-3 and raise until key phrases appear reliably. Too high and
the decoder starts forcing phrases into audio that does not contain them; the
refund mechanism prevents free-floating boost, but a sufficiently large alpha can
still outweigh genuine acoustic evidence.

Match the model's casing. English Parakeet checkpoints emit lowercase, so
lowercase the phrases for them.

## Performance

The tree is O(1) per token scored, independent of list size. Measured on one
x86 core (`-O2 -march=native`), scoring a full 1024-token vocabulary — one beam
expansion step:

| Phrases | Nodes | Build | Per expansion (1024 tokens) |
|---|---|---|---|
| 10 | 47 | 0.03 ms | 3.8 us |
| 100 | 461 | 0.08 ms | 5.7 us |
| 1 000 | 4 204 | 0.69 ms | 9.2 us |
| 20 000 | 70 858 | 15.0 ms | 8.8 us |

For a 7.4 s clip on `parakeet-tdt_ctc-110m` (~90 encoder frames) at beam 4, that
is roughly 2-3 ms of boosting for the whole utterance, against an encoder pass
of ~100 ms — under 1%, and flat from 1k to 20k phrases. Build the tree once and
reuse it; the C-API does this for you.

## CLI

```sh
# Inline phrases (repeatable), with beam search
parakeet-cli transcribe \
  --model parakeet-tdt_ctc-110m.gguf \
  --input audio.wav \
  --beam-size 4 \
  --boost "kubernetes" --boost "parakeet" \
  --boost-alpha 2.0

# From a file: one phrase per line, blank lines and '#' comments ignored
parakeet-cli transcribe \
  --model parakeet-tdt_ctc-110m.gguf \
  --input audio.wav \
  --beam-size 4 \
  --boost-file phrases.txt
```

Phrases the vocabulary cannot represent are skipped with a warning on stderr
rather than silently ignored.

### Diagnosing a phrase that does not boost

`--boost-debug` prints how each phrase was tokenized:

```sh
parakeet-cli transcribe --model parakeet-tdt_ctc-110m.gguf \
  --input audio.wav --beam-size 4 --boost "kubernetes" --boost-debug
```

```
boost phrase tokenization (1 phrase(s)):
  kubernetes                   -> ▁ku | bern | etes   [412,3201,88]
```

Compare those pieces against the token ids the model actually emits for audio
that does contain the phrase (`--json` reports per-token ids). If they differ,
the tree is waiting on a sequence the decoder never produces — see
[Limitations](#limitations) — and boosting that phrase silently does nothing.
Boosting the emitted ids directly via `pk::BoostingTree` is the workaround.

## C API

The phrase list attaches to the context and is compiled once, then reused by
every subsequent call — attach it after loading, not per utterance.

```c
parakeet_ctx* ctx = parakeet_capi_load("parakeet-tdt_ctc-110m.gguf");

const char* phrases[] = {"kubernetes", "parakeet", "jane doe"};
int accepted = parakeet_capi_set_boost_phrases(
    ctx, phrases, 3,
    /*alpha=*/2.0f, /*context_score=*/0.0f, /*depth_scaling=*/0.0f);
/* 0 for context_score/depth_scaling takes the NeMo defaults (1.0 / 2.0).
   accepted < 3 means some phrase was not tokenizable; -1 is an error. */

char* json = parakeet_capi_transcribe_path_nbest_json(
    ctx, "audio.wav", /*beam_size=*/4, /*nbest=*/1, /*score_norm=*/1, NULL);

parakeet_capi_clear_boost_phrases(ctx);   /* back to unboosted decoding */
```

ABI version 7 adds `parakeet_capi_set_boost_phrases` and
`parakeet_capi_clear_boost_phrases`. Both are additive; existing entry points
are unchanged.

## C++ API

```cpp
pk::BoostingSpec spec;
spec.phrases = {"kubernetes", "parakeet"};
spec.alpha = 2.0f;

std::vector<std::string> rejected;
pk::BoostingTree tree = model->build_boosting_tree(spec, &rejected);

pk::BoostingConfig boost;
boost.tree = &tree;          // borrowed: the tree must outlive the call
boost.alpha = spec.alpha;

auto hyps = model->transcribe_pcm_nbest(
    pcm, 16000, /*beam_size=*/4, /*nbest=*/1, /*score_norm=*/true, "", boost);
```

## Limitations

- **Phrase tokenization is greedy longest-match**, not SentencePiece's Viterbi
  unigram segmentation: the GGUF stores piece strings but not their unigram
  log-probabilities, so the exact scoring segmentation is not recoverable. This
  matches SentencePiece on ordinary words; it can differ on unusual strings,
  in which case the tree expects a tokenization the decoder may not emit.
  Callers needing exactness can build a `pk::BoostingTree` from token ids
  directly.
- **Streaming is not wired.** `pk::StreamingSession` does not take a
  `BoostingConfig` yet; `--boost` with `--stream` is rejected rather than
  silently ignored.
- **Batched decoding is not wired** (`transcribe_pcm_batch*`).
- Only one boost list per context, applied to every call on it.
