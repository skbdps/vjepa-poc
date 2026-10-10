# Independent Day5 result audit

Status: **passed**. Frozen milestone pass: **False**.

Recomputed 12,096 dense CSV rows and 1,512 parent rows directly from cached NumPy masks and regenerated labels. All source/cache/prompt/truth hashes, aggregates, sliver counts, four guards and paired bootstrap intervals match.

Per arm and representation: 3,024 part-frames = 2,670 visible + 354 absent; frame zero excluded.

Effective wrong-car pixels: 839 → 70; relative reduction 91.657%. No protected overlap pixels were released.

| Condition | Wrong-car pixels, raw → parent | Visible IoU change | Visible recall change |
|---|---:|---:|---:|
| crossing | 839 → 70 | -6.834 pp | -4.843 pp |
| long_occlusion | 0 → 0 | -0.019 pp | -0.016 pp |
| scale_camera | 0 → 0 | -0.007 pp | -0.013 pp |

The subset policy guarantees non-increasing wrong-car pixels. The measured reduction must be interpreted with its visible-detail cost. A pooled milestone pass can conceal a condition-specific regression.

- Reuses the frozen renderer; independently verifies scoring, not renderer realism.
- Cache hashes and manifests establish artifact consistency, not independent observation of model execution.
- Secondary patch readout is outside this dense-edit audit.
- This audit does not prove generalization outside the twelve fixed synthetic scenes.
