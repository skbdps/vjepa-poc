# Direct JEPA latent editing

This experiment returns to the original Day2 latent-intervention hypothesis:
change the representation of one selected object, then compare with the JEPA
representation of a video containing that exact intended change. The earlier
Day3–7 tracking work is supporting research, not evidence of latent editing.

See [PROTOCOL.md](PROTOCOL.md) for the fixed paired-video design, information
budget, controls and decision rules. **Completed:** 24 training, 8 development
and 16 held-out paired videos, with six trained correction heads and a frozen
independent coarse readout. [Results and failure diagnosis](RESULTS.md) report
support for the original copying idea and a negative result for our learned
addition: direct token copying reaches a normalized regional latent error of
0.1187 versus 0.1499 for learned correction (no edit = 1; lower is better).

The held-out comparison uses independently encoded genuine target videos.
The coarse readout places the copied object within 1.33 pixels on average,
but source-hole artifacts remain. This does not establish full video
reconstruction, automatic part discovery or propagation from one edited frame.
The [numerical audit](run_2026-10-09/analysis/independent_audit.json) reproduces
all scores, with two incomplete training-curve CSV tails documented alongside
the complete 30-epoch training log.

- [Frozen evaluation summary](run_2026-10-09/test/summary.json)
- [Per-clip scores](run_2026-10-09/test/per_clip.csv) and [per-tubelet scores](run_2026-10-09/test/per_tubelet.csv)
- [Fixed familiar-shift preview](run_2026-10-09/analysis/test_11200_dx+32_comparison.mp4)
- [Fixed unseen-shift preview](run_2026-10-09/analysis/test_11208_dx+48_comparison.mp4)
- [Reproduction notebook](Day8_Direct_Latent_Edit.ipynb) and [raw artifact checksums](run_2026-10-09/artifact_archives.json)

Preview model panels are 24×24 diagnostic probe outputs, not generated RGB videos.
The actual full run used local CPU bfloat16 after validation against a completed
Colab CPU float32 reference; Colab GPU and TPU quota was unavailable.

## Implementation

- `data.py`: deterministic matched source/target videos, pixel-level preservation
  checks and independent24/8/16train/dev/test scene seeds.
- `extract.py`: pinned official V-JEPA2.1 ViT-L/384 features in original1024D space.
- `operators.py`: full-token transport, residual transport, negative controls and
  a242,624parameter correction with a shared sequence appearance anchor.
- `train.py`: two correction arms × three paired seeds; development-only
  checkpoint selection and a completed-checkpoint freeze.
- `probe.py`: independent frozen token-only readout for coarse occupancy and RGB.
- `evaluate.py`: fixed-region latent errors, semantic readouts and scene-level
  uncertainty; does not tune models on test.

Masks of both source objects across all32frames are supplied as an explicit
oracle. This isolates representation editing from object selection and tracking.
Destination masks are derived from the requested displacement. Target features
are supervision/scoring only. The full desired trajectory is supplied; there is
no claim of causal forecasting, physical-interaction simulation, photorealistic
rendering, face consistency, automatic discovery or established method novelty.

## Execution order

Use a CUDA Colab runtime where available. CPU is supported with the same model,
using float32 inference and float16 stored features. CPU bfloat16 autocast is
an optional acceleration only after the training-fixture precision gate in
[PROTOCOL.md](PROTOCOL.md) passes: precision MSE divided by float32 edit MSE must
be below0.01 globally, in the source hole and at the destination, separately for
source and target encodings of `train_11000_dx+32`. Compare against Colab float32
with identical official weights/checksum and1024dimensional features; otherwise
use float32. Keep calibration outputs separate. Do not mix extraction backends
or precisions within the production train/dev/test cache.
Install the pinned official upstream source and
`timm==1.0.26`, `einops`, `numpy`, `pillow`, `matplotlib`, `opencv-python-headless`.

```bash
python experiments/day8/data.py --self-check
python experiments/day8/probe.py --self-check
python experiments/day8/extract.py --out /content/day8/features --splits train dev
python experiments/day8/train.py --features /content/day8/features --out /content/day8/training --device cuda
```

Commit the completed `training/checkpoint_freeze.json` before opening test. Then
verify its bytes against the committed copy and use that exact freeze:

```bash
python experiments/day8/extract.py --out /content/day8/features --splits test --freeze /content/day8/training/checkpoint_freeze.json
python experiments/day8/evaluate.py --features /content/day8/features --run /content/day8/training --out /content/day8/test --device cuda
```

For CPU, add `--device cpu` to extraction and training/evaluation. Extraction
defaults to float32; after the recorded precision gate passes, CPU extraction
may use `--device cpu --precision bfloat16`. This changes encoder autocast only;
stored features remain float16 and the edit heads/readout use float32.
Feature caches
include source/target RGB hashes, official source revision, checkpoint checksum,
extraction source hashes and numerical precision/runtime evidence. Learned
weights and feature caches are archived separately under the repository's
no-model-weights policy. A coarse readout visualization is never presented as
a generated edited video.

## Reading the results

`combined_exploratory_pass` means relative improvement in latent error and
position, with coarse appearance and distractor checks. It does not mean the
object reached the requested destination, that its old-location ghost vanished,
or that a learned model contributed beyond deterministic transport. Read the
actual centroid error in pixels, missing predictions, IoU, ghost occupancy,
temporal velocity error and appearance metrics, together with the genuine-target
reference and the per-tubelet outputs. Color identity is only coarse two-object
color discrimination, not detailed texture or face identity.

Claims about learned correction require paired scene-level comparisons with
both naive and residual transport. Claims about content inputs require the
paired geometry-only comparison too. Average the three seeds within each scene
before bootstrapping;16test scenes remain16independent observations. A primary
latent improvement alone must not be described as semantic editing success.

Development checkpoint selection uses train-fitted denominator floors,
`max(1e-8, 0.01 * median(training no-op region MSE))`; test ratios use the actual
no-op denominator with a1e-12 numerical floor. Report the development clipped
counts and test degenerate counts. These policies differ, so development and
test scores are not interchangeable. Raw MSE, no-op MSE and counts are retained.
