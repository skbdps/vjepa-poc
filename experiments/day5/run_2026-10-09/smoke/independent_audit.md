# Independent Day5 result audit

Status: **passed**. Frozen milestone pass: **False**.

Recomputed 3,024 dense CSV rows and 378 parent rows directly from cached NumPy masks and regenerated labels. All source/cache/prompt/truth hashes, aggregates, sliver counts, four guards and paired bootstrap intervals match.

Per arm and representation: 756 part-frames = 669 visible + 87 absent; frame zero excluded.

Effective wrong-car pixels: 221 → 6; relative reduction 97.285%. No protected overlap pixels were released.

| Condition | Wrong-car pixels, raw → parent | Visible IoU change | Visible recall change |
|---|---:|---:|---:|
| crossing | 221 → 6 | -10.144 pp | -10.466 pp |
| long_occlusion | 0 → 0 | -0.013 pp | -0.011 pp |
| scale_camera | 0 → 0 | -0.015 pp | -0.018 pp |

The subset policy guarantees non-increasing wrong-car pixels. The measured reduction must be interpreted with its visible-detail cost. A pooled milestone pass can conceal a condition-specific regression.

- Reuses the frozen renderer; independently verifies scoring, not renderer realism.
- Cache hashes and manifests establish artifact consistency, not independent observation of model execution.
- Secondary patch readout is outside this dense-edit audit.
- This audit does not prove generalization outside the twelve fixed synthetic scenes.
