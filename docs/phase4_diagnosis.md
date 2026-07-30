# Phase 4: Diagnosis Before Fix

## Overview

Phase 3 shipped a working baseline (`HistGradientBoostingClassifier`,
isotonic-calibrated) with ROC-AUC 0.6735 and an abstention tradeoff curve
that never finds a good operating point: hallucination-rate-of-answered
only drops from ~0.61 (answer everything) to ~0.47 across the usable
threshold range, reaching ~0.34 only once coverage has already collapsed
to near zero. Tuning that model's hyperparameters would be polishing
whichever of the two possible root causes isn't actually the problem, so
Phase 4 diagnosed first, then fixed:

1. **Step 1 (done):** clean-subset AUC -- does the model already
   discriminate well on the least-ambiguous label pairs, or does the
   ceiling hold even there? **Result: feature problem, not a label
   problem** (see below).
2. **Step 2 (superseded):** entailment-feature ablation -- planned as a
   confirmatory check, skipped once Step 1's result plus Phase 2's own
   documented caveat about the NLI framing made the diagnosis
   unambiguous enough to proceed straight to a fix.
3. **Step 3 (done):** built and wired in a query-passage reranker feature
   (`reranker.py`) to replace the three dead entailment features.
4. **Step 4 (held):** hyperparameter search + formal threshold selection.
   Blocked on re-extracting all 6000 rows with the new feature set
   (~36 min on the user's machine) and retraining -- not done in this
   session.

## Step 1: clean-subset AUC

**Question:** is the ~0.6735 ceiling a feature problem (the model can't
discriminate even on easy cases) or a label problem (`truncated_span_removed`
is a genuinely ambiguous corruption type that caps the achievable score)?

**Method:** `scripts/diagnose_subset_auc.py` loads the *existing*
`artifacts/eval/model.joblib` (no retraining) and computes ROC-AUC on
three slices of the eval set, using the fact that each Phase 1
`corruption_type` maps to exactly one label (documented in
`evaluate.py`'s `_error_by_corruption_type`):

| Subset | Rows kept | What it isolates |
|---|---|---|
| a. clean | `gold_included` vs `distractor_only` | answer clearly present vs. clearly, topically absent -- the least ambiguous pair |
| b. full | all three corruption types | Phase 3's baseline, ~0.6735 |
| c. hard | `gold_included` vs `truncated_span_removed` | the near-miss pair `error_by_corruption_type` already flagged (32.2% false-answer rate at t=0.5, ~4x `distractor_only`'s) |

**Incident along the way:** the first run crashed with
`ModuleNotFoundError: No module named '_loss'` on `joblib.load`. Cause:
`artifacts/eval/model.joblib` had been re-pickled during a prior
session's verification pass, which ran inside a Linux sandbox capped at
scikit-learn 1.7.2 (Python 3.10 there; `scikit-learn>=1.9.0` requires
Python >=3.11) -- while the user's actual venv runs 1.9.0. The pickle's
internal `_loss` module layout changed between those versions. Fix:
`retrain_and_diagnose.bat`, which re-runs `scripts/train_and_evaluate.py`
on the user's own machine (re-pickling with the venv's real 1.9.0) before
running the diagnostic -- no training code, hyperparameters, or feature
logic changed, purely a re-pickle.

**Results (real, from the user's machine):**

| Subset | ROC-AUC |
|---|---|
| a. `gold_included` vs `distractor_only` (clean) | **0.7854** |
| b. full eval set (baseline) | 0.6735 |
| c. `gold_included` vs `truncated_span_removed` (hard) | **0.5495** |

**Interpretation:** this is the feature-problem pattern, not the
label-problem pattern. On the clean pair the model does reasonably well
(0.7854) -- nowhere near the ~0.85+ that would suggest features are
basically fine and only `truncated_span_removed` needs fixing, but a
real, clearly-above-chance signal. On the hard pair, 0.5495 is barely
above chance (0.50): the model essentially cannot tell "passage with the
answer" apart from "the exact same passage with only the answer sentence
removed." That is a topical-relatedness-vs-answer-presence distinction,
and every surviving Phase 3 feature (entailment's NLI-proxy, diversity's
embedding cosine similarities, entity coverage) measures the former, not
the latter -- two passages about the same film score almost identically
on all of them whether or not the specific answer sentence is present.
Entailment was the one feature nominally positioned to catch this and
didn't (see `entailment.py`'s own module docstring: it's a semantic-
overlap proxy borrowed off-distribution from NLI, not a verified
entailment judgment, "the one design choice in the whole feature set most
likely to need revisiting if the classifier underperforms" -- which is
exactly what happened).

## Step 2: entailment-feature ablation

**Status: superseded, not run as a separate confirmatory experiment.**
Step 1's result (0.5495 on the exact case entailment was meant to
handle) combined with entailment.py's own pre-existing documented caveat
made the diagnosis clear enough to move directly to Step 3's fix rather
than spend a retrain-and-compare cycle confirming what was already well
evidenced. Phase 3's permutation importance had already shown two of the
three entailment features at negligible-to-noise-floor importance
(`docs/phase3_evaluate.md` finding 2); Step 1 supplied the mechanism
(topical-relatedness framing can't see answer-sentence removal) rather
than just the symptom.

## Step 3: the fix -- `reranker.py`

Replaced the three entailment features
(`entailment_max_prob`, `entailment_mean_prob`,
`best_margin_chunk_entailment_prob`) with two reranker features
(`reranker_max_score`, `reranker_mean_score`) from
`cross-encoder/ms-marco-MiniLM-L6-v2` -- a cross-encoder trained directly
on "does this passage answer this query" (MS MARCO passage ranking), not
NLI. See `src/abstention_features/reranker.py`'s module docstring for the
full design rationale (raw unbounded logit output, no softmax, query-
passage pair order).

**Changed:**

- **`src/abstention_features/reranker.py`** (new): lazy-singleton +
  swappable-loader pattern matching `entailment.py`/`diversity.py`.
  `entailment.py` itself is untouched and still in the codebase, just no
  longer called by the pipeline.
- **`tests/test_reranker.py`** (new): 11 fast tests against a fake
  scorer + one `@pytest.mark.slow` real-model test that checks the
  observed score range and -- the actual point of this module --
  verifies the real reranker scores a passage with the answer sentence
  intact higher than the same passage with it removed.
- **`src/abstention_features/pipeline.py`**: `feature_names` now 5 keys
  (reranker's 2 + diversity's 2 + entity_coverage's 1), down from 6.
  `extract_features`'s `entailment_model=` parameter renamed
  `reranker_model=`.
- **`tests/test_pipeline.py`**: updated fakes/assertions for the new
  5-key feature set.
- **`scripts/extract_features.py`**: loads `reranker.MODEL_NAME` instead
  of `entailment.MODEL_NAME`; `extract_split`'s `entailment_model=`
  parameter renamed `reranker_model=`.
- **`tests/test_extract_features_script.py`**: updated fake model +
  parameter names.
- **`src/abstention_model/features.py`**: no code changes needed
  (`FEATURE_COLUMNS` is derived from `pipeline.feature_names`, not
  hand-typed) -- only docstring/comment updates (7 columns -> 6).
- **`tests/test_features.py`**: fixture rows updated to reranker feature
  names/values.

**Test suite: 188 passed, 6 deselected** (up from 177/5 -- the 11 new
`test_reranker.py` fast tests + its 1 slow test).

**Not done in this session (per explicit instruction):**

- Re-extraction of all 6000 rows with the new pipeline -- **must be run
  on your machine** (`scripts/extract_features.py`, ~36 min, same CPU
  cross-encoder inference bottleneck as before) before anything below can
  happen. The existing `data/*_features.parquet` files have the *old*
  6-raw-feature schema and are now stale for this pipeline.
- Retraining the classifier on the new feature set.
- Re-running evaluation / the clean-subset AUC diagnostic against a
  reranker-trained model.
- Any change to `diversity.py`, `entity_coverage.py`, or Phase 1's data
  generation code.
- Deleting `entailment.py` -- it remains in the codebase, just
  disconnected from the pipeline.

## Step 4: hyperparameter search + threshold selection

**Status:** held, per plan -- not started. Blocked on re-extraction and
a retrain with the new feature set. The open question this step will
need to answer once unblocked: does `reranker_max_score` /
`reranker_mean_score` actually lift the `gold_included` vs
`truncated_span_removed` AUC off of 0.5495?

## Known documentation drift (not addressed this session)

`README.md`'s Phase 2 section and `scripts/demo_entailment.py` both still
describe/exercise the pre-Phase-4 entailment-based pipeline (feature
names, counts, and the "6-key feature vector" framing are now stale).
Neither was in this session's requested step list and neither is broken
(both still run -- `entailment.py` wasn't deleted), but they'll drift
further out of sync with the actual pipeline until updated. Flagging
rather than silently editing either, since neither was requested.
