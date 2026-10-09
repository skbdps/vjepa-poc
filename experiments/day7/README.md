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

## Current execution status

Implementation and fabricated-input checks are complete for the data layer and
model. The GPU training/evaluation run is being prepared. No measured improvement
or failure of this learned mechanism is claimed yet. Test scenes remain sealed
until all trained checkpoints and development selections are frozen.

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
no-model-weights policy. Curves, checkpoint hashes, raw predictions and reports
will be saved with the run.
