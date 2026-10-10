# Day 9: temporal memory improves direct JEPA latent editing

The source-hole repair worked on the fresh test. Keeping Day 8's copied
destination unchanged, the selected source-only temporal repair reduced mean
normalized hole error from **0.224936 to 0.039466**, an **82.45% reduction**.
Every one of the 16 new scenes improved on the local-fill baseline. The
predeclared primary confidence interval excluded zero, the RGB/ghost checks
passed, and the independent arithmetic audit passed.

This supports continuing the original latent-intervention direction. The new
contribution in this experiment is a custom, deterministic repair operator on
frozen JEPA features: retrieve observations of a background location elsewhere
in the source video, align their context, and use them only where the object
has been removed. No new neural head or JEPA architecture was trained, no SAM
output was used, and field-level novelty has not been established.

## What was tested

The [frozen protocol](PROTOCOL.md) used the existing eight development scenes
to select among two temporal-memory variants and five blend strengths. It
selected **aligned temporal mean, alpha = 1**, with development hole ratio
0.040844. Test scenes were new seeds 12200–12215: eight at familiar signed
32/64-pixel moves and eight at held-out signed 48/80-pixel moves. Old Day 8 test
caches were not used for selection or this evaluation.

Each operator received original-video features, the complete selected-object
and distractor mask sequences, and the requested displacement. A donor was an
observation at the same spatial location in another tubelet, clear of both
objects and their one-patch halos. The aligned variant added a feature offset
estimated from background positions visible at both times. Genuine target
features and RGB were scoring references, never repair inputs. Cells without
donors retained the exact original local fill.

Only the vacated source hole could change. The destination and every other
token remained bitwise equal to the naive copy in all scenes. This preservation
is enforced by construction; it is not evidence that the model learned object
consistency.

## Held-out measurements

The latent ratio divides each scene's edited-to-target hole MSE by its
original-to-target hole MSE, then averages scenes equally. No edit has ratio 1.
RGB is the frozen probe's coarse patch-mean prediction on a 0–1 scale; ghost
occupancy is its mean hole probability, not a count of rendered duplicate
objects.

| Method | Hole latent ratio ↓ | Hole RGB MSE ↓ | Ghost occupancy ↓ | Centroid error, px ↓ |
|---|---:|---:|---:|---:|
| Original local fill + copy | 0.224936 | 0.015194 | 0.073679 | 2.472 |
| Plain temporal memory | 0.050055 | **0.002321** | 0.004750 | 1.181 |
| Selected aligned memory | **0.039466** | 0.002589 | **0.004685** | **1.172** |
| Genuine target readout | 0 | 0.002003 | 0.002915 | 1.158 |

The primary selected-minus-naive difference was **−0.185470**, with paired
95% scene-bootstrap interval **[−0.199166, −0.171668]**. Raw hole feature MSE
fell from 1.213418 to 0.211677. The unchanged destination ratio was 0.028066;
the equal-weight hole/destination ratio improved from 0.126501 to 0.033766.

| Shift group | Scenes | Naive hole ratio | Selected hole ratio | Paired difference, 95% interval |
|---|---:|---:|---:|---:|
| Familiar magnitudes | 8 | 0.218110 | 0.040629 | −0.177481 [−0.191288, −0.165634] |
| Held-out magnitudes | 8 | 0.231762 | 0.038303 | −0.193460 [−0.214687, −0.168280] |

Mean hole RGB error decreased **82.96%** and mean ghost occupancy **93.64%**.
Their paired differences were −0.012604 [−0.015420, −0.009556] and −0.068994
[−0.081513, −0.056607], respectively. These percentages compare reported
scene means; they are not averages of individual percentage changes. All
intervals use 5,000 paired scene resamples and are exploratory, without a
multiple-comparison correction. No scored scene-arm region had a degenerate
denominator.

Alignment did **not** win every measure. Plain temporal memory had lower mean
RGB error than aligned memory, despite worse latent agreement. This secondary
comparison supports a latent-objective advantage, not universal visual
superiority or proof that the offset has isolated causal context.

## What the probe does and does not show

The genuine-target readout passed its validity gate: 1.158-pixel centroid error
and correct coarse color identity in all 256 eligible tubelets, with none
skipped. Copied/temporally repaired outputs also retained 100% coarse color
identity and had no missing selected or distractor centroids. Distractor
readout changes were exactly zero.

Selected-object IoU was **identical** for naive and both temporal repairs:
0.931623, versus 0.938913 on genuine targets. Naive hole occupancy already stayed
below the 0.5 IoU threshold. Removing lower-confidence occupancy can improve the
centroid, whose threshold is 0.25, without changing binary shape. The velocity
error declined from 2.237 to 1.540 pixels per tubelet, but the entire trajectory
was supplied and copied. This is cleaner readout of the edited sequence, not
learned motion or autonomous temporal propagation. The pointwise probe is not
a video decoder; separately encoded target-feature agreement provides the
principal independent intervention check.

## Coverage, limitations, and reproducibility

Donors existed for **3,558 of 3,630 hole tokens (98.02%)**. Averaging each
scene's coverage equally gives **98.29%**. Four scenes contained the remaining
72 cells, all retained in the primary analysis with naive fallback. Average
donor count was 6.85 per hole cell when first averaging within scenes: 1.86 from
the same encoder block and 4.99 from the other block. Two donor alignments used
the global fallback; none was rejected for lacking common background context.
These results rely substantially on offline access across the video.

The [pretest receipt](run_2026-10-09/pretest_publication.json) records publication
at commit `7bfe8d2a0da420bb5a7efdf9aee1be516b06f52b` and byte verification at
**2026-10-09 18:37:45 UTC**, before any fresh test encoding. Freeze SHA256:
`9ba4dd83b2f718b0dfaad518d9d6a9e5ad81422b27f47d99f0b1fa0dc21f4dcc`.

The run used local CPU bfloat16 inference after the Colab gateway returned
502. The restored runtime did **not** reproduce the earlier BF16 fixture
bitwise; the failed strict attempt is preserved. It did pass the unchanged
six-check comparison to the original Colab float32 reference: the largest
precision-error/edit-signal ratio was **0.008059**, below 0.01. This one-fixture
engineering check does not guarantee identical numerics everywhere. See the
[precision gate](run_2026-10-09/runtime_gate.json) and
[failed exact-repeat diagnostic](run_2026-10-09/runtime_exact_attempt.json).

The [independent audit](run_2026-10-09/analysis/independent_audit.json) passed
255,709 checks. It reconstructed all 96 full 1024-dimensional edited tensors
with matching hashes, reran the frozen readout, and reproduced 96 scene-arm,
1,536 tubelet, and 960 coverage-stratum rows, development selection, and
bootstrap decisions. It did not rerun the encoder or independently establish
GitHub publication chronology.

Full evidence: [summary](run_2026-10-09/test/summary.json),
[per-scene scores](run_2026-10-09/test/per_clip.csv),
[coverage strata](run_2026-10-09/test/coverage_strata.csv), and
[fixed held-out-shift comparison](run_2026-10-09/analysis/test_12208_dx+48_comparison.mp4).
Visuals show coarse readouts, not generated full-resolution videos.

The next step is to retain copy plus repair, then separately test propagation
from one edit anchor and robustness to camera/background motion on fresh
scenes. Full masks, static backgrounds, textured circles, and complete video
access remain strong assumptions. Faces, car parts, hidden background
generation, and real-video identity consistency are still unproven.
