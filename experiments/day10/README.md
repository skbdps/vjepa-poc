# Single-image JEPA animation proof of concept

This experiment asks whether one image, one selected-object mask, and an
explicit movement command can produce a sequence of controlled JEPA edits.
It uses V-JEPA 2.1's native image input, without a source video or temporal
background donors. The existing spatial-fill/token-copy operator and frozen
coarse readout architecture are reused; no new generative model is trained.

The [initial run](INITIAL_RESULTS.md) passed latent-distance and localization
checks but failed color reconstruction, including on genuine reference images.
The [native-image readout follow-up](native_readout_PROTOCOL.md) trains the same
small probe on disjoint still-image data and reserves new images for testing.
The JEPA encoder and edit operator remain unchanged. Both runs are retained.

See the [fixed protocol](PROTOCOL.md) and [results](RESULTS.md). Four fresh
images each have five nonzero movement commands plus an unchanged reference.
Correct target images are independently encoded scoring references only.
The preview is an animation of a 24×24 diagnostic readout, not reconstructed
full-resolution video. The return loop repeats existing positions and adds
no test observations.

## What the code receives

The editor receives one RGB image, one binary selected-object mask, and a list
of horizontal displacements. The native encoder receives `[1,3,1,384,384]` and
returns `[1,576,1024]`; the image is not repeated into an artificial video.
All benchmark inference, cached features, edits, and probe evaluation use FP32.

The source is encoded once. A local spatial estimate fills the selected region,
then the original selected tokens are copied to each requested destination.
The estimate is computed once and reused. This is mathematically the existing
Day8 naive operator on a single image, not a new background-repair method.
Zero displacement returns the exact source. Every position starts from that
same immutable source, so revisiting a position gives exactly the same features.

No target RGB/features, hidden background, renderer state, second-object mask,
or extra source frame enters the operator. The synthetic benchmark obtains its
one selection mask from the renderer; automatic object discovery is not tested.
Target and distractor masks appear only in scoring.

## Reproduce the environment

Run commands from this repository's root. The recorded CPU environment is
Python 3.12.14, PyTorch 2.9.0+cpu, torchvision 0.24.0+cpu, NumPy 2.3.5,
timm 1.0.26, einops 0.8.2, Pillow 12.3.0, huggingface_hub 0.36.0,
safetensors 0.6.2, and PyYAML 6.0.3. PNG/MP4 previews also require the `ffmpeg`
and `ffprobe` executables. No GPU is required for this bounded test.

```bash
python3.12 -m venv .venv-day10
source .venv-day10/bin/activate
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 timm==1.0.26 einops==0.8.2 Pillow==12.3.0 huggingface_hub==0.36.0 safetensors==0.6.2 PyYAML==6.0.3

export DAY10_UPSTREAM="$PWD/external/vjepa2"
export DAY10_ARTIFACTS="$PWD/artifacts/day10"
export DAY10_RELEASE="$PWD/experiments/day10/run_2026-10-10"
export DAY10_RUN="$PWD/reproductions/day10"
export TORCH_HOME="$DAY10_ARTIFACTS/torch"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$DAY10_ARTIFACTS" "$DAY10_RUN" external
git clone https://github.com/facebookresearch/vjepa2.git "$DAY10_UPSTREAM"
git -C "$DAY10_UPSTREAM" checkout 204698b45b3712590f06245fbfba32d3be539812
```

The loader checks the exact upstream revision and official
`vjepa2_1_vitl_dist_vitG_384.pt` checkpoint. It downloads that public checkpoint
if absent, approximately 5.15 GB, into
`$TORCH_HOME/hub/checkpoints/`. Its SHA-256 must be:

```text
7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6
```

Reuse `training/probe.pt` from the archived `Day8_Trained_Operators.zip`;
do not train or select a new probe on these images. For example, after placing
that archive in `$DAY10_ARTIFACTS`:

```bash
unzip "$DAY10_ARTIFACTS/Day8_Trained_Operators.zip" 'training/probe.pt' -d "$DAY10_ARTIFACTS/day8"
export DAY10_PROBE="$DAY10_ARTIFACTS/day8/training/probe.pt"
sha256sum "$DAY10_PROBE"
```

The required probe SHA-256 is
`bf18fc7be331d56d9aafd3462111d3f9dac622e9177a3e3e1bfd91710ab07eab`.
Day8's learned correction heads and training feature caches are not needed.

## Reproduce the initial failed readout-transfer run

The separate calibration fixture uses seed 13000, frame 15, with a 48-pixel
target displacement. It checks the native image interface and frozen readout
localization. Its IoU remains a descriptive measurement; passing localization
does not establish a faithful pixel decoder.

```bash
python experiments/day10/animate.py self-check
python experiments/day10/calibrate.py \
  --upstream "$DAY10_UPSTREAM" \
  --probe "$DAY10_PROBE" \
  --out "$DAY10_ARTIFACTS/calibration_reproduction"
```

Keep this new calibration output separate from the published calibration.
Recorded runtimes and file hashes can differ on another machine. The published
freeze binds the original calibration report and exact numerical sources;
overwriting its report would correctly fail validation.

To reproduce the fixed four-image benchmark into a new directory:

```bash
python experiments/day10/animate.py benchmark \
  --upstream "$DAY10_UPSTREAM" \
  --probe "$DAY10_PROBE" \
  --freeze "$DAY10_RELEASE/freeze.json" \
  --publication "$DAY10_RELEASE/pretest_publication.json" \
  --out "$DAY10_RUN" \
  --device cpu --threads 4

python experiments/day10/audit_results.py \
  --run "$DAY10_RUN" \
  --freeze "$DAY10_RELEASE/freeze.json" \
  --publication "$DAY10_RELEASE/pretest_publication.json" \
  --probe "$DAY10_PROBE" \
  --out "$DAY10_RUN/analysis/independent_audit.json"

python experiments/day10/visualize.py --run "$DAY10_RUN"
```

The publication receipt attests to the original pretest run. Reusing it here
reproduces that frozen experiment; it does not turn these already disclosed
scenes into a new independent validation set. Modified methods need a new
protocol, new scene seeds, and a new pretest freeze.

The benchmark resumes completed images only when their recorded files and run
metadata match. It keeps every image and position, including failures. It
excludes the zero-displacement state from normalized edit averages, averages
positions within each image, then weights the four images equally. These are
descriptive results, without confidence intervals.

## Stored evidence and restoration

The archive index accompanying the results records raw archive names, sizes,
and checksums. Extract raw Day10 evidence into one run directory, preserving
its relative paths. Do not add another enclosing directory between that run
directory and `manifest.json`. The expected layout is:

| Relative path | Contents |
| --- | --- |
| `manifest.json` | Fixed specifications, provenance, and per-image file hashes |
| `inputs/image_13200/` through `image_13203/` | One source PNG and selected-mask PNG per image |
| `targets/image_13200/` through `image_13203/` | Genuine target PNGs, restricted to scoring/display references |
| `features/cache/image_13200.npz` through `image_13203.npz` | Source/target FP32 features and scoring labels |
| `edited/image_13200.npz` through `image_13203.npz` | Copy-and-repair/wrong-direction tensors and fixed background |
| `predictions/image_13200.npz` through `image_13203.npz` | Coarse occupancy/RGB predictions and display ordering |
| `metrics/`, `invariants/` | Per-image scores and exact transport checks |
| `per_position.csv`, `per_image.csv`, `summary.json` | All 96 arm-position rows and descriptive aggregates |
| `invariants.json` | Exact checks including repeated display positions |

With those files restored, the audit and visualization commands above do not
need the encoder checkpoint or re-encoding. The audit needs the exact frozen
Day8 probe. It independently reconstructs edits, readouts, scores, and gates;
it does not independently repeat encoder inference or the remote publication
chronology. Keep the published freeze, calibration report, and publication
receipt at their original relative locations.

## Reproduce the native-image readout follow-up

The fixed training protocol uses 32 training scenes (64 genuine images), eight
development scenes (16 genuine images), seed 1010, and the final checkpoint
after 40 epochs. Development is a pass/fail gate, never checkpoint selection.
Fresh test seeds are 13600–13603. Original test seeds 13200–13203 are not used
for fitting, normalization, development or the new test.

Restore the native readout archive listed in
[the archive index](native_readout_run_2026-10-10/artifact_archives.json) over a
copy of `experiments/day10/native_readout_run_2026-10-10`. Keep its relative
paths: `training/probe.pt`, `features/cache/`, and `fresh_test/`. The Git copy
supplies reports, hashes, source/target images and visual artifacts; the archive
supplies the raw caches, readout checkpoint and predictions.

```bash
export DAY10_NATIVE_ROOT="$DAY10_ARTIFACTS/native_readout"
export DAY10_NATIVE_PROBE="$DAY10_NATIVE_ROOT/training/probe.pt"

python experiments/day10/audit_native_readout.py \
  --run "$DAY10_NATIVE_ROOT" \
  --freeze "$DAY10_NATIVE_ROOT/training_freeze.json" \
  --publication "$DAY10_NATIVE_ROOT/training_publication.json" \
  --out "$DAY10_NATIVE_ROOT/training/independent_audit_reproduction.json"

python experiments/day10/audit_native_results.py \
  --run "$DAY10_NATIVE_ROOT/fresh_test" \
  --freeze "$DAY10_NATIVE_ROOT/test_freeze.json" \
  --publication "$DAY10_NATIVE_ROOT/test_publication.json" \
  --probe "$DAY10_NATIVE_PROBE" \
  --out "$DAY10_NATIVE_ROOT/fresh_test/analysis/independent_audit_reproduction.json"

python experiments/day10/visualize.py --run "$DAY10_NATIVE_ROOT/fresh_test"
```

Those commands use caches and need no encoder inference. To re-encode the
published test using the exact archived probe:

```bash
python experiments/day10/native_readout.py test \
  --out "$DAY10_NATIVE_ROOT/fresh_test_reproduction" \
  --upstream "$DAY10_UPSTREAM" \
  --probe "$DAY10_NATIVE_PROBE" \
  --freeze "$DAY10_NATIVE_ROOT/test_freeze.json" \
  --publication "$DAY10_NATIVE_ROOT/test_publication.json"
```

`native_readout.py make-training-freeze`, `prepare`, and `train` implement the
complete fixed training pipeline. They require a verified pretraining receipt;
the test requires a second freeze binding the final probe and passing
development gate. A retrained checkpoint may differ bytewise and cannot be
silently substituted under the original test freeze. Replaying disclosed
scenes is reproduction, not new independent validation.

## Apply the operator to a supplied image

Prepare an RGB image exactly 384×384 pixels and a matching grayscale PNG mask
containing only 0 and 255. White pixels select the object. The CLI does not
resize inputs, infer a mask, or choose motion. Displacements must be integer
multiples of 16 pixels. This first CLI rejects paths for which either the
requested translation or its wrong-direction control would crop the selected
patch support; leave enough room on both sides.

```bash
python experiments/day10/animate.py image \
  --image /absolute/path/source.png \
  --mask /absolute/path/selected_mask.png \
  --shifts 0 16 32 48 64 80 \
  --upstream "$DAY10_UPSTREAM" \
  --probe "$DAY10_NATIVE_PROBE" \
  --out "$DAY10_ARTIFACTS/my_image" \
  --device cpu --threads 4

python experiments/day10/visualize.py --run "$DAY10_ARTIFACTS/my_image"
```

This produces edited latent tensors and an optional diagnostic preview. The
reusable Python API is `animate_image(encoder, source_rgb, selected_mask,
shifts_px, probe=None, device='cpu')`; omit the probe to obtain features only.

Use the archived native-image probe for this route; the initial video-trained
probe is retained only to reproduce the disclosed failed transfer.

**Both probes were trained on controlled synthetic ball scenes. Their
outputs on photographs are unvalidated and must not be treated as an image
decoder.** The method does not recover genuinely hidden background, predict
natural motion, identify faces or parts, handle occlusion/rotation, or generate
photorealistic video. Exact feature copying and repeated-position consistency
are construction guarantees, not proof of detailed visual identity.
