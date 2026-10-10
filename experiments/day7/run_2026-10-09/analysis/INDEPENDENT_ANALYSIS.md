# Day7 independent audit and predictive-supervision contrast

All saved prediction arrays were independently scored and matched the runner's per-target CSV and aggregate results.

The primary contrast averages predictive minus retrieval-only utility over the three predetermined training seeds. Every bootstrap draw samples the same18 whole clips across all seeds. These are18 independent clips, not54 videos.

| Primary difference | Mean effect (pp) | Paired clip95% interval (pp) | Seed sample SD (pp) |
|---|---:|---:|---:|
| balanced_utility | -0.008 | [-3.118, +4.131] | 0.779 |
| visible_localization | +0.242 | [-1.305, +1.612] | 2.676 |
| absent_specificity | -0.257 | [-6.607, +8.638] | 4.019 |

| Training seed | Utility difference (pp) | Visible difference (pp) | Absent-specificity difference (pp) |
|---|---:|---:|---:|
| 1701 | -0.590 | +0.363 | -1.544 |
| 1702 | -0.310 | +2.856 | -3.475 |
| 1703 | +0.877 | -2.492 | +4.247 |

Prespecified predictive-addition evidence rule: **not passed**.
The rule requires positive mean utility with an interval excluding zero, plus no more than1pp mean loss in visible localization or absent specificity. These are point-estimate guards, not formal noninferiority confidence bounds.

| Method | Visible localization | False presence | Wrong car | Immediate recovery | ID switches |
|---|---:|---:|---:|---:|---:|
| global_full | 46.21% | 5.02% | 8.83% | 22.22% | 68 |
| selected_day4 | 49.12% | 10.04% | 8.15% | 16.67% | 37 |
| global_projected_centered | 39.30% | 12.36% | 6.96% | 27.78% | 56 |
| retrieval_only_seed1701 | 91.12% | 0.39% | 4.31% | 5.56% | 42 |
| retrieval_only_seed1702 | 92.99% | 5.02% | 4.00% | 5.56% | 66 |
| retrieval_only_seed1703 | 93.77% | 5.02% | 3.48% | 11.11% | 53 |
| predictive_seed1701 | 91.48% | 1.93% | 4.36% | 5.56% | 45 |
| predictive_seed1702 | 95.85% | 8.49% | 2.75% | 50.00% | 47 |
| predictive_seed1703 | 91.28% | 0.77% | 4.41% | 5.56% | 32 |
| coordinate_only_seed1701 | 38.58% | 1.93% | 2.23% | 0.00% | 19 |
| coordinate_only_seed1702 | 38.99% | 8.49% | 3.37% | 0.00% | 18 |
| coordinate_only_seed1703 | 35.62% | 8.11% | 1.66% | 0.00% | 8 |

Future-target cosine averaged over three training seeds:

| Arm | Learned forecast | Immutable-anchor copy | Last-retrieved copy |
|---|---:|---:|---:|
| retrieval_only | -0.0638 ± 0.2779 | 0.7348 ± 0.0000 | 0.7713 ± 0.0029 |
| predictive | 0.8661 ± 0.0036 | 0.7348 ± 0.0000 | 0.7261 ± 0.0205 |

Full per-seed effects, sample standard deviations, condition breakdowns, coordinates/frozen-baseline contrasts and saved future-latent diagnostics are in independent_analysis.json.

Future-latent diagnostics compare the learned forecast with copying the immutable anchor and last retrieved representation. They summarize saved CSV values, not an independent recomputation of the feature tensors. Low latent error alone does not establish correct tracking.

The video always shows crossing seed10200 and training seed1701 across the same four fixed methods. It is not selected by outcome.

This is supervised prediction over frozen JEPA features in the same2-D procedural family. Encoder attention is offline within16-frame blocks. The result does not establish real-video/face identity, generative editing, 3-D reasoning, or self-supervised discovery.
