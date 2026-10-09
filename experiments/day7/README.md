# Learned persistent-part JEPA prototype

This is the active research direction. It trains a new part-conditioned latent
predictor and recurrent identity state on top of V-JEPA 2.1. It does not use SAM2.
The previous segmentation/editor work is retained as historical comparison and
supporting infrastructure, not as a substitute for the central hypothesis.

The 871,810-parameter head maintains an initial identity anchor, a learned memory
for each tagged part, sibling context, and a forecast consumed by future tracking.
The decisive ablation compares the same trained architecture with and without
future-latent supervision. A position-only control tests whether synthetic motion
patterns explain apparent gains. Three paired training seeds are retained.

See [PROTOCOL.md](PROTOCOL.md) for the frozen split, objectives, success guards,
and temporal/annotation limitations, and [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)
for primary-source context. This is supervised training of a custom predictive
head over a frozen backbone. It is not end-to-end JEPA pretraining, automatic part
discovery, or an established literature-novel method.

## Completed result

All nine heads completed training on a Colab Tesla T4 and were evaluated on 18
unseen synthetic clips after the development checkpoint freeze was committed.
Visible localization was 46.21% for frozen JEPA matching, 92.63% for the trained
retrieval-only head and 92.87% with predictive supervision. The predictive
addition's balanced-score effect was −0.008 percentage points, with a paired
95% interval of [−3.118, +4.131]: no demonstrated added tracking benefit.

The forecast learned its future-feature target, but recovery after occlusion is
still fragile. All 43 missed predictive recoveries across repeated seed/event
evaluations were non-emissions. See [RESULTS.md](RESULTS.md) for the complete
evidence and [PROGRESS.md](PROGRESS.md) for the completed checkpoint. The next
research target is visibility reopening while preserving part identity.

The clean [Colab notebook](Predictive_Part_JEPA_Colab.ipynb) provides the full
workflow. `analyze.py` independently recomputes tracking metrics from saved
predictions. `diagnose.py` performs separate, post-test interventions on forecast
inputs and sibling context; these are descriptive reliance checks, not additional
selected methods.

## Workflow

1. Extract frozen JEPA train/development tokens with `extract.py`; fit centering
   only on the 36 training clips.
2. Train all arms/seeds with `train.py --stage train`, using nine development clips
   only for epoch/threshold selection.
3. Commit the completed `checkpoint_freeze.json` before opening held-out clips.
4. Extract the 18 test clips using the validated freeze, then run
   `train.py --stage test` and the independent result analysis.

All test-time inputs are RGB-derived tokens and frame-zero part annotations.
Later labels are restricted to training losses or evaluation. Learned weights are
preserved separately from this repository's source/results, following its existing
no-model-weights policy. [Compact evidence](run_2026-10-09/compact_evidence.zip)
contains all scores, cells, eligibility, CSV rows, curves, reports and source.
The full archive preserves all nine weights and original internal predictions;
its SHA-256 and replay requirements are recorded in the export manifest.
