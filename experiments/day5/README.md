# Parent constraints for persistent part edits

Day 5 tests whether a separately tracked whole car can prevent a tagged door or window edit from painting the other car. Both arms share the same SAM2 child predictions. The parent arm intersects them with their independently tracked car; original overlap protection remains in place.

Read [PROTOCOL.md](PROTOCOL.md) for the frozen seeds, outcomes, limitations and success criterion. Day 4 source and results remain unchanged. This is a new-seed experiment in the same synthetic 2-D renderer, with additional parent annotations and inference cost.

## Run in Colab

[Open Parent_Constraint_Colab.ipynb](https://colab.research.google.com/github/skbdps/vjepa-poc/blob/experiment/part-consistency-colab/experiments/day5/Parent_Constraint_Colab.ipynb). Select a T4 GPU and run the cells in order.

The notebook pins experiment revision `de419ad916e57e04271c14fd830a8a52022b764e`, verifies the existing freeze, prepares the official SAM2.1 Tiny source/checkpoint, runs three smoke clips and all twelve test clips, displays the reports and crossing diagnostic, and exports a checksum-inventoried ZIP. The archive includes prediction caches, manifests, CSVs, reports, videos and frozen source; it excludes synthetic frames and model weights.

The notebook has empty outputs and has not been separately executed end to end on a clean GPU runtime. It packages the commands used by the research Colab; recorded run artifacts are the evidence for execution and results.

## Command line

Use a CUDA environment with compatible PyTorch and torchvision, Python, NumPy, Pillow, OpenCV and ffmpeg. Check out the pinned revision before running:

```bash
git checkout de419ad916e57e04271c14fd830a8a52022b764e
python experiments/day5/parent_benchmark.py --self-test
python experiments/day5/run_parent_experiment.py --self-test
```

Prepare the official model using the same helper as the research run:

```python
import sys
sys.path.insert(0, "experiments/day4")
import real_video
checkpoint = real_video.prepare_sam2(
    root="/content/sam2", checkpoint_dir="/content/checkpoints")
print(checkpoint)
```

Run the fixed stages, substituting your paths if outside Colab:

```bash
python -u experiments/day5/run_parent_experiment.py \
  --stage smoke --out /content/day5_parent_run \
  --checkpoint /content/checkpoints/sam2.1_hiera_tiny.pt \
  --frozen-path experiments/day5/frozen_config.json

python -u experiments/day5/run_parent_experiment.py \
  --stage test --out /content/day5_parent_run \
  --checkpoint /content/checkpoints/sam2.1_hiera_tiny.pt \
  --frozen-path experiments/day5/frozen_config.json
```

Do not regenerate `frozen_config.json` or change the policy after viewing test results. Repeating the same stage reuses complete clip caches after checking source, inputs, prompts and model hashes. A partially inferred clip may rerun. Complete all twelve test clips regardless of intermediate outcomes.

## Read the outputs

- `test/RESULTS.md` and `test/results.json`: aggregate, per-condition and per-clip results; raw masks, effective edits, parent accuracy, sliver diagnostics, paired clip bootstrap intervals and fixed success assessment.
- `test/dense_rows.csv`: every child part-frame, arm and representation; 3,024 scored part-frames per arm/representation, excluding frame zero.
- `test/parent_rows.csv`: independently predicted whole-car masks, including wrong-owner paint.
- `test/patch_rows.csv`: secondary Day 4 patch readout at the unchanged `0.9404296875` threshold; these decisions do not gate edits.
- `clips/*/children/sam2_masks.npz`, `clips/*/parents/sam2_masks.npz` and manifests: cached model predictions and provenance.
- `test/test_crossing_6200_comparison.mp4`: fixed first-crossing diagnostic; magenta identifies paint on the other car's full visible body.

The primary improvement is strictly fewer absolute wrong-car effective paint pixels. A pass additionally requires no more than one percentage point loss in visible mean IoU and pooled recall for both raw masks and effective edits. An intersection can suppress an unsafe edit; it cannot reconstruct a missing part or recover a lost identity. No face, generative-video or real-video generalization claim follows from this test.
