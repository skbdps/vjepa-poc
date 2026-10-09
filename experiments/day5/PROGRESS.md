# Day 5 completed execution checkpoint

The source and configuration were committed at `de419ad916e57e04271c14fd830a8a52022b764e` before any Day 5 GPU predictions. Source digest: `fea39b4e91e3e02177503b1f3c0444c54f62966e8cf4a9844ae76833cbfc6314`. Frozen experiment files remain unchanged.

All three smoke clips and twelve held-out clips completed on a Tesla T4. The execution manifest records completion at `2026-10-09T05:25:47.575017+00:00`. The full test contains 3,024 scored child part-frames per arm/representation: 2,670 visible and 354 absent. No method selection or parameter change followed smoke or test outcomes.

The independent audit passed for both stages. It verified provenance and regenerated every test dense row (12,096 across the two arms and two representations), all 1,512 parent rows, aggregate/sliver metrics, success guards and paired bootstrap intervals from saved predictions.

**Finding:** wrong-car effective paint decreases 839 → 70 pixels, but 122,251 correctly covered pixels are removed. Effective visible mean IoU declines 93.03% → 90.50% and pooled recall 93.07% → 91.30%; all four frozen accuracy guards fail. Keep parent containment off by default. See [RESULTS.md](RESULTS.md), [measured JSON](run_2026-10-09/test/results.json), [independent audit](run_2026-10-09/independent_audit.md), and the [original GPU archive](run_2026-10-09/gpu_run.zip).

The new two-car/four-part hierarchy editor and native recipe replay reuse cached predictions without another model run. The editor/replay helpers are separate from frozen experiment source. Fabricated CPU/JavaScript checks pass, including strict controls validation, original overlap protection, disabled-part protection and matched compositing arithmetic. Browser validation also passed: settings persisted across owner switching, actual recipe JSON downloads matched all four stable IDs, reset cleared edits, pasted JSON restored settings including containment ON, invalid schema import left state unchanged, playback reached frame 63 and scrubbing reached frame 18. Both downloaded OFF/ON recipes replayed across 64 frames each with all protection invariants passing. Final exported state has containment OFF. [EDITOR_VALIDATION.md](EDITOR_VALIDATION.md) records the evidence; PNG export was not verified.

Research notebook: https://colab.research.google.com/drive/1wipvFdtlDOuRwp6Cd9sQpQUCZQNxo2HP

Runtime output was `/content/day5_parent_run`. [Parent_Constraint_Colab.ipynb](Parent_Constraint_Colab.ipynb) supplies pinned reproduction instructions and an optional editor step; it has empty outputs and was not separately executed end to end in a clean GPU runtime. The actual experiment commands ran in the research notebook.

One bounded [post-hoc other-parent veto](../day6/POSTHOC_DAY5_EXPLORATION.md) was evaluated on these completed caches as development data and also failed all four guards. No new GPU inference was used, and the candidate did not warrant a fresh validation run. Day 5 remains unchanged.

Remaining packaging work: preserve the executed notebook and visual preview, and commit the completed artifacts. Do not modify the frozen policy in response to these test outcomes.
