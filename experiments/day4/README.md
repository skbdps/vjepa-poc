# Persistent video parts and selective editing

This stage builds on the Day 3 localization diagnostic. The practical prototype initializes a front-door panel and front window once, preserves their IDs, and applies an edit to a selected predicted part throughout a video. The edit is deterministic recoloring, not generative video synthesis.

## Try the working control layer

Download either self-contained HTML editor and open it in a browser. It needs no GPU, server, or network connection:

- [Car-roundabout interactive editor](run_2026-10-09/car_roundabout/interactive_part_editor.html)
- [Car-shadow interactive editor](run_2026-10-09/car_shadow/interactive_part_editor.html)
- [Two-part video preview](run_2026-10-09/car_roundabout/dual_part_edit.mp4)

Select a tagged part to adjust its independent color, strength, and enabled state. Enabling the window preserves the door's settings. Play or scrub the clip, show predicted outlines, save a frame, or export the controls as JSON. Importing that recipe restores the same settings; a wrong sequence or part registry is rejected. Reset disables all edits.

For model execution, open [Part_Consistency_Colab.ipynb](Part_Consistency_Colab.ipynb) in Colab and select a T4 GPU. The notebook downloads the pinned model and source frames, runs both clips, saves masks and scores, and builds the editors. Its repository copy has empty outputs; the executed research history is in the separate live Colab notebook below.

## What is implemented

- A 64-frame synthetic benchmark with prolonged occlusion, crossing cars, and scale/camera transformations. Separate calibration, development, and test seeds; four tagged parts per clip.
- Frozen V-JEPA global matching and a tracker combining initial appearance, context, validated motion, and optional appearance updates. Component ablations and a strong multiscale RGB-template comparator.
- A SAM2.1 Tiny specialist comparison on the same synthetic videos, including dense-mask accuracy and hidden-part hallucinations.
- Two real DAVIS car sequences with independently authored sparse part polygons. Only frame-zero masks initialize each tracker.
- SAM2 mask propagation, per-part affine flow, and a stronger shared-plane homography/SIFT comparator.
- A reusable edit API that selects a stable part ID and protects all unselected predicted masks, including overlap pixels. Saved masks allow new colors and part choices without another model run.
- Independent simultaneous part controls, portable JSON recipes, and native-frame recipe replay. Ambiguous overlaps between predicted parts remain unchanged, even if both parts are enabled.
- A standalone offline interactive editor built from the actual tracked masks.

## Run records

The code and intermediate freezes are committed as the experiment progresses on `experiment/part-consistency-colab`. The original Day 3 code and results remain in `../day3/`.

- [Protocol declared before new tests](PROTOCOL.md)
- [V-JEPA calibration/development freeze](run_2026-10-09/frozen_config.json)
- [Development results](run_2026-10-09/development_results.json)
- [First real-car results](run_2026-10-09/car_roundabout/results.json)
- [First door-edit video](run_2026-10-09/car_roundabout/door_recolor.mp4)
- [Second real-car results](run_2026-10-09/car_shadow/results.json)
- [Results and interpretation](RESULTS.md)
- [Independent result audit](AUDIT.md)
- [Visual comparison of both real clips](run_2026-10-09/two_clip_part_tracking_comparison.png)
- [Footage/model attribution](ATTRIBUTION.md)

The executed notebook is saved in [Google Colab](https://colab.research.google.com/drive/1wipvFdtlDOuRwp6Cd9sQpQUCZQNxo2HP).

## Reproduce the synthetic V-JEPA experiment

Use a CUDA Colab runtime. Keep its matching PyTorch/torchvision installation, and install `timm==1.0.26`, `einops`, `pillow`, `numpy`, and `opencv-python-headless`. `ffmpeg` must be available for preview videos.

Clone the official `facebookresearch/vjepa2` repository into `/content/vjepa2` and check out `204698b45b3712590f06245fbfba32d3be539812`. The runner loads frozen `vjepa2_1_vit_large_384`, points the pinned source at Meta's public checkpoint endpoint, and uses FP16 inference. No model training is performed.

From this repository's root:

```bash
python experiments/day4/benchmark.py --self-test
python experiments/day4/run_experiment.py --stage development --out /content/day4_run --upstream /content/vjepa2
# Preserve frozen_config.json before the next command.
python experiments/day4/run_experiment.py --stage test --out /content/day4_run --upstream /content/vjepa2
```

The runner checks a digest of its source, tracker, and benchmark before test execution. Changing them invalidates the development freeze. Features are cached per RGB content and pinned model/source identity. The cached features are deliberately excluded from the small GitHub run records; they can be regenerated.

## Reproduce the real-video prototype

SAM2 code is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4`; use the official SAM2.1 Hiera Tiny checkpoint and `configs/sam2.1/sam2.1_hiera_t.yaml`. `real_video.prepare_sam2()` installs that revision with the optional CUDA extension disabled and downloads the checkpoint. The recorded run uses FP32, appropriate for its Tesla T4.

```bash
python experiments/day4/real_video.py download --out /content/car-roundabout --sequence car-roundabout --frames 64
python experiments/day4/real_video.py run --frames-dir /content/car-roundabout/frames --annotations experiments/day4/car_roundabout_annotations.json --out /content/real_roundabout --checkpoint /content/sam2.1_hiera_tiny.pt
python experiments/day4/real_video.py download --out /content/car-shadow --sequence car-shadow --frames 40
python experiments/day4/real_video.py run --frames-dir /content/car-shadow/frames --annotations experiments/day4/car_shadow_annotations.json --out /content/real_shadow --checkpoint /content/sam2.1_hiera_tiny.pt
```

`run()` supplies only `initial_masks(annotations, shape)` to every tracker. Later polygon frames are read only by the scorer. Input frame hashes and original URLs are in `car_roundabout_source_manifest.json` and `car_shadow_source_manifest.json`.

Replay an edit from saved masks:

```bash
python experiments/day4/part_editor.py --frames-dir /content/car-roundabout/frames --masks /content/real_roundabout/sam2_masks.npz --target-id 1 --color 35 140 240 --out /content/door_blue.mp4
python experiments/day4/part_editor.py --frames-dir /content/car-roundabout/frames --masks /content/real_roundabout/sam2_masks.npz --target-id 2 --color 240 180 35 --out /content/window_amber.mp4
```

Replay simultaneous edits from an editor-exported recipe:

```bash
python experiments/day4/replay_recipe.py \
  --frames-dir /content/car-roundabout/frames \
  --masks experiments/day4/run_2026-10-09/car_roundabout/sam2_masks.npz \
  --registry experiments/day4/run_2026-10-09/car_roundabout/part_registry.json \
  --recipe experiments/day4/run_2026-10-09/car_roundabout/dual_part_edit.recipe.json \
  --out /content/dual_part_edit.mp4
```

The recipe stores controls only. Native replay uses original input JPEG pixels; the browser preview uses a compressed embedded MP4, so those rendered pixels are not expected to be identical. Both paths use the same native-resolution masks and protect overlapping predictions.

## Reproduce the specialist synthetic comparison

```bash
python experiments/day4/sam2_benchmark.py --self-test
python experiments/day4/sam2_benchmark.py --stage calibration --out /content/day4_sam2_run --checkpoint /content/sam2.1_hiera_tiny.pt
# Preserve sam2_frozen_config.json before the next command.
python experiments/day4/sam2_benchmark.py --stage test --out /content/day4_sam2_run --checkpoint /content/sam2.1_hiera_tiny.pt
```

The specialist was added after the V-JEPA development run and the first real-video demonstration. Its algorithm and mask-to-patch readout were fixed before its test. It uses the same synthetic scenes and frame-zero annotations, with different model objectives and temporal processing. High-quality JPEG conversion for SAM2 is measured and reported. Dense-mask results are separate from the calibrated patch-localization readout.

## Interpretation limits

V-JEPA sees future frames inside each independent 16-frame window; it is not a causal video encoder in this test. Synthetic cars are 2-D sprites, not 3-D views. The real polygons are approximate assistant-authored diagnostics, not official DAVIS part labels or human-verified ground truth. The real sequences may overlap SAM2's training data and do not contain full target occlusion. They demonstrate a pipeline, not unseen-video generalization.

Outside-mask preservation is exact on the edited arrays before lossy encoding and relative to the predicted masks. A wrong mask can still recolor the wrong real-world pixels. No face consistency, novel-view synthesis, arbitrary replacement, or full generative-video editing has been established.
