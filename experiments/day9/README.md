# Source-hole repair with temporal JEPA memory

Day8 supported the original latent-copying idea but left the vacated object
location imperfect. Day9 preserves the copied destination exactly and tests
whether original-video background observations can repair that hole.

The new operators use only original JEPA tokens, full source-object/distractor
masks, and the requested translation. They never receive target or empty-scene
features. Plain temporal memory averages eligible same-position background
tokens. Context-aligned memory first adjusts donor features using positions
clear in both query and donor times. This is a hypothesis about contextual
features, not an established factorization or a new JEPA architecture.

See [PROTOCOL.md](PROTOCOL.md) for input limits, selection, numerical gates and
the fresh 16-scene test. Development selected aligned temporal memory with
alpha=1: mean hole/no-op error 0.04084 versus 0.20830 for Day8 local fill.
These are development results only; held-out evaluation is pending.

## Run order

Restore the archived Day8 development features and frozen probe checkpoint.
Use the pinned official V-JEPA source/weights and CPU BF16 path described in the
protocol. Validate this runtime against the archived Colab FP32 training fixture.
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

This remains a static-background synthetic, offline, full-trajectory editing
test. The frozen 24×24 readout is a coarse diagnostic, not a generated RGB video.
New test scenes are seeds12200–12215; none of the old Day8 test clips is used for
selection or substituted for this fresh validation.
