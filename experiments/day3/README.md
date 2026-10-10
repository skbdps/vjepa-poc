# Day 3: persistent object-part diagnostic

Question: after one initial annotation, do frozen V-JEPA features keep identifying the same part of the same object?

This small synthetic experiment uses two cars, each with a front-door panel and a window. Corresponding parts share their appearance; body context distinguishes the cars. The conditions are motion, partial occlusion, and full occlusion followed by reappearance. This first diagnostic does not test camera rotations, realistic faces, automatic semantic naming, generative editing, or a complete tracker.

## Protocol

- 3 calibration clips and 6 evaluation clips with disjoint seeds.
- 32 RGB frames per clip at 384 x 384, encoded as two **disjoint** 16-frame windows.
- Frozen V-JEPA 2.1 ViT-L, initial annotated patch prototypes, global cosine matching.
- Later masks and visibility are used only for calibration/scoring, never to guide matching.
- Absence thresholds use calibration clips only, with balanced presence accuracy.
- Compare RGB patch matching, fixed initial location, and OpenCV pyramidal Lucas-Kanade.
- Report exact denominators, visible-part localization, wrong-car/part matches, false presence, occlusion recovery, and results within/across encoded windows.
- Score at the actual 16 x 16 spatial patches / two-frame tubelet resolution. Ambiguous target visibility is excluded and counted. An emitted ambiguous/background location on an otherwise visible target is an error.

Within each window, the video encoder can use later frames. These are **offline video features**, not causal prediction. Across-window results test reuse of initial prototypes without shared encoder attention across windows. Passing this tiny diagnostic would justify wider evaluation; failure would identify a limitation of this matching method, not invalidate the overall project.

## Run

Use a Colab CUDA GPU runtime, keeping its preinstalled PyTorch/torchvision. Install `timm==1.0.26`, `einops`, `numpy`, `pillow`, and `opencv-python-headless` if absent. Clone the official `facebookresearch/vjepa2` source and check out `204698b45b3712590f06245fbfba32d3be539812`.

```bash
python experiments/day3/part_consistency.py --self-test --out-dir /tmp/day3_self_test
python experiments/day3/run_experiment.py --upstream /content/vjepa2 --out /content/day3_run
```

The runner sets the pinned source's checkpoint URL to Meta's official public endpoint in memory. It uses CUDA FP16 autocast, checks shape/finite values, and records model, source revision, environment, seeds and peak GPU allocation. Feature caches use a content hash and are safe to resume. No training or Drive mount is required.

## Output

`manifest.json`, `results/results.json`, per-prediction `results/predictions.csv`, two annotated comparison videos, source inputs and reusable frozen feature caches. Preserve the run folder outside Colab's temporary runtime before disconnecting.

The oracle fixtures in `--self-test` validate only the evaluator, not V-JEPA performance. Real model results are reported only after pretrained inference finishes.

## Completed Colab run (2026-10-09 IST)

The replayable `VJEPA_Part_Consistency_Day3.ipynb` is included without cell outputs, following the repository policy. The executed outputs remain in Colab. It pins the experiment code to commit `461dd0264dfe023bc9639479a388fa2eda785692`. The live notebook is [available in Colab](https://colab.research.google.com/drive/1wipvFdtlDOuRwp6Cd9sQpQUCZQNxo2HP#scrollTo=2qUgyGzS0SBh). Aggregate results and environment metadata are in `run_2026-10-09/`.

| Metric | V-JEPA | Lucas-Kanade optical flow |
| --- | ---: | ---: |
| Correct visible localization | 276/315 (87.6%) | 276/315 (87.6%) |
| False presence while fully hidden (lower is better) | 6/36 | 36/36 |
| Correct at first visible tubelet after full absence | 2/4 | 0/4 |
| Second independently encoded window | 147/167 (88.0%) | 129/167 (77.2%) |

The run used a Tesla T4 and peaked at 1.338 GiB of GPU allocation during inference. No training was performed. The comparison is a small diagnostic with six evaluation clips and only four recovery events. Cars remain in separate lanes with a fixed camera. Optical flow has no explicit occlusion/reidentification logic and receives frame 0, while V-JEPA initialization uses the first two-frame tubelet. These results do not establish real-video, face, camera-angle, or generative-editing consistency.
