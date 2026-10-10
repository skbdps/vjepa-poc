# Source-hole repair with temporal JEPA memory

**Completed: source-only temporal repair improved the vacated object location
on all 16 fresh scenes while preserving the copied destination exactly.**
The development-selected aligned memory reduced mean normalized source-hole
error from **0.224936 to 0.039466 (82.45%)**. Its paired difference from Day8
local filling was −0.185470, with 95% scene-bootstrap interval
[−0.199166, −0.171668]. Mean hole ghost occupancy fell 93.64% and coarse hole
RGB error fell 82.96%. These percentages compare scene means.

The frozen selection was aligned temporal memory with alpha=1, chosen on eight
old development scenes before the fresh test. Both familiar and held-out shift
groups improved. Plain temporal memory had worse latent agreement but slightly
better RGB error than aligned memory; alignment does not win every metric.

The new operators use only original JEPA tokens, full source-object/distractor
masks, and the requested translation. They never receive target or empty-scene
features. Plain temporal memory averages eligible same-position background
tokens. Context-aligned memory first adjusts donor features using positions
clear in both query and donor times. This is a hypothesis about contextual
features, not an established factorization or a new JEPA architecture.

Read [RESULTS.md](RESULTS.md) for the complete outcome, interpretation, and
limitations, and [PROTOCOL.md](PROTOCOL.md) for the rules frozen before testing.
The [independent audit](run_2026-10-09/analysis/independent_audit.json) passed
255,709 checks, including all 96 reconstructed full feature tensors, frozen
readouts, development selection, and scene-bootstrap summaries.

Completed evidence:

- [Scores and decision checks](run_2026-10-09/test/summary.json),
  [per-scene measurements](run_2026-10-09/test/per_clip.csv), and
  [donor/fallback strata](run_2026-10-09/test/coverage_strata.csv).
- [Fixed held-out-shift comparison video](run_2026-10-09/analysis/test_12208_dx+48_comparison.mp4)
  and [visual audit](run_2026-10-09/analysis/visual_audit.json).
- [Pretest publication receipt](run_2026-10-09/pretest_publication.json),
  [runtime precision gate](run_2026-10-09/runtime_gate.json), and
  [archive names, sizes, and checksums](run_2026-10-09/artifact_archives.json).

Donors covered 3,558 of 3,630 hole tokens (98.02%). The other 72 tokens in four
scenes used exact naive fallback and remained in the primary result. This is a
static-background synthetic, offline, full-trajectory editing test. No new
neural head was trained, and the frozen 24×24 readout is a coarse diagnostic,
not a generated full-resolution RGB video. Binary object IoU did not change;
lower ghost probabilities improved centroid estimates without learning motion.

## Run order

Restore the archived Day8 development features, the original training fixture
`train_11000_dx+32`, their original feature manifest, the original Colab FP32
fixture, and the frozen probe checkpoint. These reused inputs come from the
saved Day8 feature/reference/checkpoint archives; the index above identifies
the newly saved Day9 evidence.
Use the pinned official V-JEPA source/weights and CPU BF16 path described in the
protocol, with `TORCH_HOME` pointing to the checkpoint cache. The completed run
used PyTorch 2.9.0+cpu and NumPy 2.3.5. Validate a restored runtime against the
archived Colab FP32 training fixture before encoding test scenes.
The restored environment passed all six original 1% precision limits despite
not being bitwise identical to the previous CPU; both checks are retained.

```bash
python experiments/day9/temporal.py --self-check
python experiments/day8/validate_precision.py --reference /path/to/colab/features/cache/train_11000_dx+32.npz --upstream /path/to/vjepa2 --outdir /path/to/day9/precision --threads 8
python experiments/day9/runtime_gate.py --precision-dir /path/to/day9/precision --old-features /path/to/day8/features --out /path/to/day9/runtime_gate.json
python experiments/day9/evaluate.py --mode dev --features /path/to/day8/features --probe /path/to/day8/training/probe.pt --out /path/to/day9/dev
```

Publish the completed protocol/source/development selection freeze and verify
its exact bytes before fresh test extraction. The saved freeze and publication
attestation describe this run; a new run must publish and verify its own freeze.
The extractor rejects missing or changed frozen dependencies and precision evidence.

```bash
python experiments/day9/extract.py --out /path/to/day9/features --upstream /path/to/vjepa2 --reference /path/to/day8/features/manifest.json --freeze /path/to/day9/freeze.json --publication /path/to/day9/pretest_publication.json
python experiments/day9/evaluate.py --mode test --features /path/to/day9/features --probe /path/to/day8/training/probe.pt --out /path/to/day9/test --freeze /path/to/day9/freeze.json
```

Reconstruct and audit the numerical evidence, then render the two fixed scene
comparisons and summary charts:

```bash
python experiments/day9/audit_results.py --features /path/to/day9/features --test /path/to/day9/test --freeze /path/to/day9/freeze.json --publication /path/to/day9/pretest_publication.json --probe /path/to/day8/training/probe.pt --dev-features /path/to/day8/features --out /path/to/day9/analysis/independent_audit.json
python experiments/day9/visualize.py --test /path/to/day9/test --out /path/to/day9/analysis
```

This completed test uses seeds 12200–12215; no old Day8 test clip was used for
selection or substituted for fresh validation. Repeating these seeds reproduces
this result. Any further method development informed by these results requires
a newly declared holdout. The next research steps are single-anchor edit
propagation and camera/background-motion tests while retaining successful
copy-and-repair behavior.
