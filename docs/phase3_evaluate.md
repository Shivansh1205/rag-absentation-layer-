# Phase 3: Classifier, Calibration, and the Abstention Tradeoff Curve

## Overview

`src/abstention_model/evaluate.py` computes the abstention tradeoff curve --
for a sweep of confidence thresholds, how much the classifier abstains,
how often it hallucinates-by-proxy on the rows it does answer, and how
much of the genuinely answerable eval set it still covers -- plus the
standard supporting diagnostics (ROC-AUC, PR-AUC, a confusion matrix,
permutation feature importance, and an error breakdown by Phase 1's
`corruption_type`). `scripts/train_and_evaluate.py` is the thing that
actually runs it: load Phase 2's feature parquets, fit the baseline
`HistGradientBoostingClassifier` (`train.py`), calibrate it
(`calibrate.py`), evaluate it (`evaluate.py`), and write every artifact
to disk. `evaluate.py` itself does no file I/O -- it's pure computation
returning DataFrames/dicts, by design, so its logic can be unit-tested
without touching a filesystem.

## How to run

```bash
uv run python scripts/train_and_evaluate.py
```

Defaults: `--data-dir data` (expects `train_features.parquet` /
`eval_features.parquet` there), `--output-dir artifacts/eval`,
`--calibration-cv 5`. `artifacts/eval/` is reserved for evaluation
outputs specifically -- Phase 4 tuning is expected to write its own
`artifacts/tune/` or `artifacts/final/` alongside it, not into this
same directory.

## Output artifacts

Written to `artifacts/eval/` by `scripts/train_and_evaluate.py`:

- **`model.joblib`** -- the calibrated classifier (`CalibratedClassifierCV`
  wrapping a `HistGradientBoostingClassifier`, isotonic, `cv=5`). This is
  the object whose `predict_proba` every threshold decision in this
  document is computed from -- not `train.py`'s plain baseline fit,
  which is run but not persisted separately.
- **`calibration_curve.png`** / **`calibration_curve.csv`** -- predicted
  probability vs. observed label=1 frequency (`prob_pred`, `prob_true`,
  `n_per_bin`, 10 quantile bins). Both are built from `evaluate()`'s own
  `"calibration_curve"` output, so the plot and the table are guaranteed
  to show the same numbers rather than being two independent
  recomputations.
- **`tradeoff_curve.png`** -- `abstention_rate`,
  `hallucination_rate_of_answered`, and `coverage_of_answerable`, all
  plotted against threshold.
- **`threshold_sweep.csv`** -- the same curve as a full 101-row table
  (`threshold` from 0.00 to 1.00 in steps of 0.01), including the raw
  counts (`n_answered`, `n_answered_and_hallucinated`,
  `n_answered_and_correct`, `n_abstained_from_answerable`) behind every
  rate, so the numbers are inspectable without recomputing anything.
- **`feature_importance.csv`** -- permutation importance
  (`scoring="roc_auc"`, `n_repeats=20`) computed on the *eval* set, not
  train, sorted descending.
- **`error_by_corruption_type.csv`** -- false-abstain / false-answer
  rates per `corruption_type` at the default threshold.
- **`summary.json`** -- the headline scalars in one machine-readable
  file: `default_threshold`, `roc_auc`, `pr_auc_positive_class_1`,
  `pr_auc_positive_class_0`, `confusion_matrix` (labels + matrix + the
  "not a claimed optimum" note), `calibration_signed_mean_deviation`,
  `calibration_abs_mean_deviation`, and the top 4
  `top_permutation_importance_features`.

## Real results (5000 train / 1000 eval)

- **ROC-AUC: 0.6735**
- **PR-AUC, label=1 (answerable) positive: 0.5283**
- **PR-AUC, label=0 (unanswerable / hallucination-risk) positive: 0.7595**

The gap between the two PR-AUC numbers is expected, not a red flag on
its own: average precision's no-skill baseline is the positive class's
prevalence, and label=0 is the majority class in eval (~61% vs. label=1's
~39%). A higher baseline floor for label=0 means its PR-AUC reads higher
for a comparable level of actual discrimination -- the two numbers are
answering different questions ("how well can this rank answerable
questions to the top" vs. "how well can this rank hallucination-risk
questions to the top"), not directly comparable to each other.

Top 4 permutation importances (eval set):

| Feature | Importance (mean) |
|---|---|
| `entity_coverage_fraction` | 0.0670 |
| `centroid_question_relevance_cosine` | 0.0659 |
| `chunk_redundancy_mean_cosine` | 0.0195 |
| `entailment_max_prob` | 0.0181 |

## Key findings

**1. Relevance and entity coverage are co-dominant, not relevance-dominant.**
The design hypothesis going into Phase 3 was that
`centroid_question_relevance_cosine` would dominate feature importance.
The real numbers don't support that: `entity_coverage_fraction` (0.0670)
and `centroid_question_relevance_cosine` (0.0659) are within noise of
each other (importance std ~0.0125 and ~0.0131 respectively) at the top
of the ranking. Half the hypothesis holds (relevance matters a lot);
the "dominant" framing doesn't -- coverage matches it, it doesn't trail it.

**2. Entailment features contribute near-zero signal at t=0.5.**
`entailment_max_prob` (0.0181) and `best_margin_chunk_entailment_prob`
(0.0065) are a distant third/fourth tier, and `entailment_mean_prob`
came back essentially at noise floor (~-0.0001 -- permuting it did not
measurably hurt the model). `entity_coverage_undefined` is similarly
negligible (0.0012). These four are the deletion candidates for the next
tuning iteration: cutting them would shrink the feature set with, on
this evidence, minimal cost to ROC-AUC.

**3. At the default threshold t=0.5, the model false-abstains on 64.2%
of genuinely answerable questions (`gold_included` group). Threshold
selection is not optional -- 0.5 is not a neutral operating point.**
At `t=0.5`, `error_by_corruption_type` shows a 64.2% false-abstain rate
on `gold_included` rows -- the majority of genuinely answerable eval
questions get incorrectly abstained on at the "default" threshold. On
the hallucination-risk side, `truncated_span_removed`'s false-answer
rate (32.2%) is roughly 4x `distractor_only`'s (8.2%) -- confirming the
harder, near-miss corruption type is the real driver of
hallucination-risk errors, not the easy topically-unrelated distractors.
Both facts point the same direction: `t=0.5` was never a considered
choice, just a placeholder, and it behaves like one.

## Limitations

- Hyperparameter tuning was deliberately deferred (`train.py` uses
  `HistGradientBoostingClassifier` defaults plus `class_weight="balanced"`
  and a fixed `random_state` only) -- these are baseline numbers, not a
  tuned model's numbers.
- `model.joblib` is a `joblib`/pickle artifact of a fitted
  `CalibratedClassifierCV`. `scikit-learn` does not guarantee pickle
  compatibility across versions, even minor ones -- `pyproject.toml` now
  pins `scikit-learn>=1.9.0,<1.10.0` (the exact version confirmed
  installed in this project's venv) specifically to reduce the risk of
  a future `uv sync` producing an environment that can't load a model
  saved by an earlier one.

## Next steps

- **Phase 4 (tuning):** hyperparameter search on
  `HistGradientBoostingClassifier`, informed directly by finding 2 above
  -- worth trying the entailment-feature-pruned set as one of the search
  candidates, not just tuning tree parameters on the full 7-feature set.
- **Calibration diagnostic:** originally a one-off check, now a
  first-class part of every `train_and_evaluate.py` run
  (`calibration_curve.csv`/`.png` + the deviation numbers in
  `summary.json`). The last run found a small net overconfidence (signed
  mean deviation -0.0214) and a modest absolute calibration error
  (0.0413), with the roughest patch of the curve sitting right around
  `prob_pred` ~0.44-0.56 -- i.e. exactly where `t=0.5` sits. Isotonic
  calibration is doing a reasonable job overall, but combined with
  finding 3, the region right around the default threshold is both
  miscalibrated-relative-to-the-rest-of-the-curve *and* known to
  false-abstain heavily -- another reason threshold selection belongs in
  Phase 4's scope rather than staying at 0.5.
