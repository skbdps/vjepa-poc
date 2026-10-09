# Parent constraints for persistent part edits

Day 5 tests whether a separately tracked whole car can prevent a tagged door or window edit from painting the other car. Both arms share the same SAM2 child predictions. The parent arm intersects them with their independently tracked car; original overlap protection remains in place.

Read [PROTOCOL.md](PROTOCOL.md) for the frozen seeds, outcomes, limitations and success criterion. Day 4 source and results remain unchanged. This is a new-seed experiment in the same synthetic 2-D renderer, with additional parent annotations and inference cost.

## Completed finding

**Keep containment off by default.** The complete, independently audited twelve-clip test reduces wrong-car effective paint **839 → 70 pixels**, but loses **122,251 correctly covered pixels**. Effective visible IoU falls **93.03% → 90.50%** and pooled recall **93.07% → 91.30%**. All four frozen accuracy guardrails fail; the large relative wrong-car reduction starts from an already tiny absolute area after existing overlap protection. The largest regressions occur during crossing and on thin visible parts.

Read [RESULTS.md](RESULTS.md) for exact denominators, raw-mask versus effective-edit metrics, condition-level regressions, uncertainty and cost. [Test JSON](run_2026-10-09/test/results.json), [independent audit](run_2026-10-09/independent_audit.md), and the [original complete GPU archive](run_2026-10-09/gpu_run.zip) preserve the evidence. Both smoke and test finished without policy changes. The four-control hierarchy artifact passed browser interaction and actual browser-export → native-replay checks in both containment modes; see [EDITOR_VALIDATION.md](EDITOR_VALIDATION.md). PNG export was not verified.

One bounded [Day 6 post-hoc other-parent veto](../day6/POSTHOC_DAY5_EXPLORATION.md) also failed all four guards on these same caches used as development data. It did not justify a new GPU validation run and is not a fresh held-out comparison.

## Run in Colab

[Open Parent_Constraint_Colab.ipynb](https://colab.research.google.com/github/skbdps/vjepa-poc/blob/experiment/part-consistency-colab/experiments/day5/Parent_Constraint_Colab.ipynb). Select a T4 GPU and run the cells in order.

The notebook pins experiment revision `de419ad916e57e04271c14fd830a8a52022b764e`, verifies the existing freeze, prepares the official SAM2.1 Tiny source/checkpoint, runs three smoke clips and all twelve test clips, displays the reports and crossing diagnostic, and exports a checksum-inventoried ZIP. The archive includes prediction caches, manifests, CSVs, reports, videos and frozen source; it excludes synthetic frames and model weights.

The notebook has empty outputs and has not been separately executed end to end on a clean GPU runtime. It packages the commands used by the research Colab; recorded run artifacts are the evidence for execution and results.

After all twelve test clips finish, an optional cell fetches the separately pinned editor/replay helper revision `c55f436f9abce93b35b511e67f41e1140f271445`, verifies that the frozen experiment files still match, builds the standalone hierarchy editor and displays it in Colab. The measured experiment remains pinned to `de419ad`; the later helper revision only creates replay artifacts. The exported run manifest records both roles explicitly, and the archive preserves the helper sources under `artifact_helpers/` separately from frozen experiment source. Set `BUILD_EDITOR = False` to skip that cell's work. This helper cell shares the notebook's clean-runtime execution caveat above.

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

## Hierarchy editor and replay without a GPU

The editor exposes two cars and four independently controlled parts. Its stable car/part identities are assigned from frame-zero annotations, not automatically discovered. Recipes contain only enabled flags, RGB colors, strengths, sequence identity and the containment switch; they contain no model weights, executable code or masks.

Build the standalone editor after the completed run is available:

```bash
python experiments/day5/build_hierarchy_demo.py \
  --run-root /path/to/day5_parent_run \
  --scene test_crossing_6200 \
  --out /path/to/hierarchy/interactive_hierarchy_editor.html
```

Saved-run artifact locations are [interactive_hierarchy_editor.html](run_2026-10-09/hierarchy/interactive_hierarchy_editor.html), [scene_registry.json](run_2026-10-09/hierarchy/scene_registry.json), and [browser_replay.mp4](run_2026-10-09/hierarchy/browser_replay.mp4). The editor includes its own compressed preview and predicted masks and can run offline after download. The registry records stable hierarchy IDs, frame-zero annotations and source hashes. These are replay artifacts, not an additional inference method.

After the complete test results and all twelve prediction caches are downloaded, export a JSON recipe from the hierarchy editor and run:

```bash
python experiments/day5/replay_hierarchy.py \
  --run-root /path/to/day5_parent_run \
  --scene test_crossing_6200 \
  --recipe /path/to/browser_export.recipe.json \
  --out /path/to/hierarchy_edit.mp4
```

This reuses saved child and parent masks and regenerates the exact synthetic source RGB after checking completed-run provenance. It writes the MP4, a normalized `.recipe.json`, and a per-frame `.audit.json`. No model executes. Every part keeps its stable string ID, independent color, strength and enabled state; the global containment toggle adds the predicted-parent intersection. Disabled parts and original child-mask overlaps remain protected in both modes.

The CLI imports the editor builder's validator and compositor, so exported controls use one schema and one native implementation. Both use `floor((1-strength)*RGB + strength*color)` per permitted channel. Browser preview uses compressed video RGB; native replay uses original rendered RGB, so their actual images need not be bit-identical. Protection checks apply before lossy MP4 encoding.

Run `python experiments/day5/replay_hierarchy.py --self-test` for fabricated CPU-only behavior checks. `--default-recipe --scene test_crossing_6200` prints a controls template without loading or generating the scene.
