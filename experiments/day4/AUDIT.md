# Day 4 result audit

Audited 2026-10-09 from the saved V-JEPA calibration/development freeze, the completed V-JEPA and SAM2 18-clip synthetic test records, both real-car run records, and the inference/scoring source. SAM2 test results below are from the held-out split; its calibration results are kept separate.

## Validity checks

- The local V-JEPA benchmark/tracker/runner source digest exactly matches the development freeze: `aae772368b244728f8085a1ed3ddf1967872ead4c4321a0ae1bfc17e26199b00`. The selected method is `no_memory`, selected using the declared development objective, with calibration-only thresholds.
- The SAM2 comparator source matches its calibration freeze: `491bb936ed608d3ceeb34f1ed316425143ffed669ee35a2a9a4add9cf28c5ee4`. Its mask-to-patch readout and threshold are frozen before its test. The threshold is maximum predicted patch coverage, not a calibrated model probability.
- Inference receives RGB/features and frame-zero annotations. Later target masks enter calibration or scoring, not the tracking entrypoints. This checks the code's information boundary; files alone cannot independently establish when an annotator viewed predictions.
- Independent aggregation of all 13,392 synthetic test CSV rows reproduced visible localization, hidden false presence, and recovery numerators/denominators for all six methods. Saved paired localization intervals were also reproduced.
- SAM2 adds 2,232 test CSV rows with exactly the same evaluation keys and labels as each V-JEPA/baseline method. Independent CSV aggregation reproduced its patch outcomes, all dense report sums from 4,536 part-frame rows, and all 33 paired metric estimates and bootstrap intervals in the final comparison. The calibration and test freezes are identical as parsed JSON, the CSV threshold is exactly `0.9404296875`, and all 18 test manifests match the frozen source/checkpoint/readout and frame-zero-only prompts. Every saved mask archive hash matches its manifest.
- Both real-car sequences' scores were recomputed from all three saved tracker-mask arrays and exactly matched the reported tracking and selective-mask scores.
- No material inference or scoring bug was identified in these checks. No frozen inference or scoring file was changed for this audit.

## Synthetic test outcome

There are 18 clips, six per condition, with four parts and 64 frames per clip. The patch task scores 1,922 visible and 260 fully absent target-tubelets after initialization; 50 ambiguous target-tubelets are excluded. A tubelet contains two frames. The first four frames are stationary by design; excluding the first tubelet still leaves one stationary scored tubelet per target.

| Test metric | Selected V-JEPA (`no_memory`) | Initial global V-JEPA | Template + flow |
|---|---:|---:|---:|
| Visible localization | 1,026/1,922 (53.38%) | 1,064/1,922 (55.36%) | 1,215/1,922 (63.22%) |
| False presence when fully hidden | 14/260 (5.38%) | 53/260 (20.38%) | 63/260 (24.23%) |
| First-reappearance localization | 5/18 (27.78%) | 7/18 (38.89%) | 0/18 (0%) |
| Wrong car on visible targets | 183/1,922 (9.52%) | 221/1,922 (11.50%) | 26/1,922 (1.35%) |
| Defined car-identity transitions | 60 | 93 | 18 |
| Mean of visible localization and absent specificity | 74.00% | 67.49% | 69.49% |

The selected V-JEPA method trades visible localization for fewer false reports during absence. It does not demonstrate uniformly better tracking than the classical comparator. Appearance-memory updates were not selected, so this run does not establish that the proposed online memory helps the chosen objective.

The following differences are selected V-JEPA minus template, using the same 2,000 paired whole-clip bootstrap draws and seed 914 as the run. False-presence differences favor V-JEPA when negative. The false-presence, recovery, balanced-objective, and condition intervals are supplementary post-run analysis; no algorithm was changed using them.

| Difference | Estimate, percentage points | Descriptive 95% interval |
|---|---:|---:|
| Overall visible localization | -9.83 | [-16.03, -3.71] |
| Overall hidden false presence | -18.85 | [-27.48, -3.67] |
| First-reappearance localization | +27.78 | [+6.67, +47.62] |
| Balanced objective | +4.51 | [-3.53, +9.85] |

The balanced-objective interval includes zero; it does not establish a clear overall advantage over template + flow. Against initial global V-JEPA, the selected method changes visible localization by -1.98 points [-3.91, +0.26] and hidden false presence by -15.00 points [-23.05, -7.48].

| Condition, six clips each | Selected visible localization | Template visible localization | Paired difference interval, points |
|---|---:|---:|---:|
| Crossing | 271/702 (38.60%) | 430/702 (61.25%) | [-25.25, -19.77] |
| Long occlusion | 413/476 (86.76%) | 416/476 (87.39%) | [-6.90, +5.00] |
| Scale/camera | 342/744 (45.97%) | 369/744 (49.60%) | [-14.11, +5.78] |

Crossing is a substantial unresolved identity failure: selected V-JEPA reports the wrong car in 173/702 visible checks (24.64%), versus zero for template + flow. Both methods miss all six crossing reappearance checks. In long-occlusion clips, visible localization is similar, while selected V-JEPA has 14/248 hidden false reports versus 63/248 and recovers on 5/12 first-reappearance checks versus 0/12. Scale/camera clips contain no fully absent targets.

Visible localization for selected V-JEPA over windows 1–4 is 77.52%, 57.45%, 36.48%, and 42.61%; template + flow gives 76.26%, 51.19%, 50.77%, and 72.73%. These windows contain different motion/occlusion phases, so the decline is a diagnostic, not a controlled estimate of a window-boundary effect.

Bootstrap samples preserve whole clips rather than treating correlated frames as independent. They describe variation under this small renderer family, not real-video generalization. Six clips per condition and 18 reappearance events remain limited evidence. Zero-observed-error bootstrap intervals can collapse to zero and do not prove zero future risk. Intervals are descriptive and are not corrected for multiple comparisons.

## Completed SAM2 specialist test

SAM2.1 Tiny was evaluated with the frozen calibration threshold and mask-to-patch readout. The comparison script validates all 18 held-out clip names, all 2,232 target-tubelet keys per method, identical ground-truth labels, consistent boolean/outcome fields, and the frozen V-JEPA thresholds. SAM2's source and threshold freeze were checked separately. Neither model selection nor threshold fitting is performed by the comparison.

| Common patch metric | SAM2.1 Tiny | Selected V-JEPA (`no_memory`) | Template + flow |
|---|---:|---:|---:|
| Visible localization | 1,826/1,922 (95.01%) | 1,026/1,922 (53.38%) | 1,215/1,922 (63.22%) |
| False presence when fully hidden | 11/260 (4.23%) | 14/260 (5.38%) | 63/260 (24.23%) |
| First-reappearance localization | 6/18 (33.33%) | 5/18 (27.78%) | 0/18 (0%) |
| Wrong car on visible targets | 55/1,922 (2.86%) | 183/1,922 (9.52%) | 26/1,922 (1.35%) |
| Defined car-identity transitions | 4 | 60 | 18 |

Differences below are SAM2 minus the named comparator, with 2,000 paired whole-clip bootstrap draws and seed 914. Values and intervals are percentage points. Negative hidden false-presence differences favor SAM2.

| Comparator | Visible localization difference [95% interval] | Hidden false-presence difference [95% interval] | Immediate recovery difference [95% interval] |
|---|---:|---:|---:|
| Selected V-JEPA | +41.62 [+30.94, +50.10] | -1.15 [-5.80, +3.41] | +5.56 [-12.50, +27.78] |
| Initial global V-JEPA | +39.65 [+30.08, +47.19] | -16.15 [-26.52, -5.88] | -5.56 [-33.33, +25.00] |
| Template + flow | +31.79 [+22.60, +40.09] | -20.00 [-27.06, -8.60] | +33.33 [+11.11, +57.14] |

The specialist has substantially better visible patch localization on this renderer family. The hidden-presence and immediate-recovery intervals against selected V-JEPA include zero, so this test does not establish a difference on those two outcomes. The earlier V-JEPA-versus-template tradeoff is unchanged by adding SAM2.

By condition, SAM2 visible localization is 638/702 (90.88%) on crossing, 444/476 (93.28%) on long occlusion, and 744/744 (100%) on scale/camera. All 55 wrong-car checks and all four defined identity transitions occur in crossing; its 2/6 immediate recoveries and long-occlusion's 4/12 remain weak despite high overall localization. Scale/camera contains no absence or recovery events. The wrong-car check count and transition count measure different things: a sustained wrong identity can generate many wrong-car checks after one switch.

### Separate raw dense-mask readout

Dense scores use every source frame after frame zero and count any nonempty ground-truth part as visible. They apply no calibrated patch-presence gate. There are 3,997 visible and 539 absent part-frames, totaling 4,536 across 18 clips; these are not the patch table's 1,922 visible and 260 absent two-frame target-tubelets. The dense table includes thin visibility excluded as ambiguous from the patch readout.

| Dense SAM2 metric | Test result |
|---|---:|
| Mean visible part-frame IoU | 94.42% over 3,997 visible part-frames |
| Pooled visible-pixel precision | 9,884,807/10,491,554 (94.22%) |
| Pooled visible-pixel recall | 9,884,807/10,394,102 (95.10%) |
| Pooled precision including predictions on hidden targets | 9,884,807/10,566,570 (93.55%) |
| Nonempty raw mask when the part is absent | 60/539 (11.13%) |
| Predicted pixels on absent parts | 75,016 |

Thus the 4.23% patch false-presence rate cannot be described as the raw dense-mask false-presence rate; the latter is 11.13% under a different temporal unit, visibility rule, and ungated readout. Dense mean IoU is 91.13% on crossing, 94.00% on long occlusion, and 97.92% on scale/camera. No cross-method dense-IoU comparison is possible here because the V-JEPA benchmark predicts patch locations rather than masks.

Full counts, all six SAM2-versus-baseline comparisons, paired intervals, validation flags, and input hashes are saved in `run_2026-10-09/comparison/comparison.json`; a compact table is in `comparison.md`. These descriptive synthetic intervals do not remove the small-sample, renderer-family, training-objective, JPEG-input, temporal-processing, or compute mismatch limitations stated below.

## Supplementary recovery timing

The frozen recovery metric measures only the first unambiguous visible tubelet. A separate `recovery_analysis.py` was proposed after V-JEPA test results but before SAM2 held-out scores were viewed, then applied to saved CSV predictions without model reruns or changes to frozen scoring. Its horizon-zero counts exactly reproduce every method's primary result. This is a post-hoc descriptive outcome, not a replacement primary outcome.

Events come only from existing recovery labels. Latency is the offset in source-frame indices between two-frame tubelets; it is not exact per-frame timing, wall-clock latency, or causal detection delay. A new fully absent tubelet ends the episode, and missing rows or clip end truncate follow-up. Ambiguous tubelets cannot be hits and are recorded explicitly. The implementation was checked with fabricated trajectories containing delayed recovery, intervening ambiguity, renewed absence, missing rows, and clip-end censoring.

| Horizon after first visible tubelet | Fully observed events | Selected V-JEPA observed successes | Global V-JEPA observed successes | Template observed successes | SAM2 observed successes |
|---|---:|---:|---:|---:|---:|
| 0 source frames | 18/18 | 5/18 | 7/18 | 0/18 | 6/18 |
| 2 source frames | 18/18 | 11/18 | 12/18 | 0/18 | 8/18 |
| 4 source frames | 18/18 | 12/18 | 12/18 | 2/18 | 8/18 |
| 8 source frames | 6/18 | 12 known | 12 known | 16 known | 8 known |

At the eight-frame horizon, 12 events reach clip end before the complete horizon. A hit observed before clip end remains a known success. Selected V-JEPA has one unresolved censored event, giving 12–13 possible successes among 18; global V-JEPA and template have no unresolved events because their censored trajectories already contain hits. Among the six fully observed eight-frame horizons, selected V-JEPA recovers on 1/6, global V-JEPA on 0/6, and template on 4/6. No intervening ambiguous or renewed-absence tubelets occur in these actual evaluated follow-up paths; the analyzer nevertheless handles and reports both.

The template's immediate 0/18 score therefore does **not** mean it cannot recover: it often recovers later. Saved details are in `run_2026-10-09/vjepa_recovery_analysis.json`, including complete-horizon denominators, every event trajectory, censor reasons, and exact immediate-metric verification.

Applying the same supplementary analyzer to the completed SAM2 CSV exactly reproduces its primary 6/18 immediate count. SAM2 reaches 8/18 by two source frames and remains at 8/18 by four. At eight source frames, 12 events reach clip end early: six already recovered and six are unresolved. This gives eight known successes and an 8–14/18 bound allowing unknown censored outcomes; among the six complete eight-frame horizons, SAM2 recovers on 2/6. There are no ambiguous follow-up tubelets or renewed absences in the evaluated paths. These results are in `run_2026-10-09/sam2_recovery_analysis.json` and do not replace the primary recovery metric.

## Real-car demonstration

The real-video methods are SAM2.1 Tiny and classical geometry/flow; these are not real-video V-JEPA results. Named part IDs are supplied in the first-frame annotation and retained by the pipeline, rather than discovered semantically by the model.

| Sequence | Sparse part checks | SAM2 mean part IoU | LK affine | Shared homography |
|---|---:|---:|---:|---:|
| Car-roundabout, 64 frames | 16 | 86.18% | 42.01% | 74.64% |
| Car-shadow, 40 frames | 10 | 81.48% | 64.52% | 52.10% |

The roundabout permitted door-edit mask has 91.94% IoU, 93.62% macro precision, and 98.14% macro recall over eight annotated frames. It includes 4,375 pixels outside the approximate door polygons among 67,673 permitted edit pixels (93.54% pooled precision). Its predicted door/window masks do not overlap in any of the 64 frames, so protective subtraction does not change this run's door mask. Shadow's permitted door-mask IoU is 80.79% over five annotated frames.

For shadow, SAM2's permitted door-mask precision is 86.07% and recall is 92.68%. Protection removes 32 overlapping predicted-mask pixels across all 40 frames. Nevertheless, the permitted edit contains 1,537 pixels outside the approximate door polygons across five scored frames, including 14 annotated window pixels at frame 32. The polygons have a stated four-pixel boundary uncertainty, so these small overlaps are diagnostic rather than definitive anatomical ground truth; they still illustrate why prediction-relative protection is not an anatomical guarantee.

“Selective edit” measures the permitted predicted mask against the door polygon. It is not an independent measure of rendered edit quality. The recoloring implementation guarantees that pre-encoding array changes remain in the selected predicted mask and outside other protected predicted masks. Wrong masks can still recolor wrong physical pixels; lossy video encoding also limits exact preservation claims about the exported MP4.

Both real sequences use sparse approximate assistant-authored polygons, not official DAVIS part ground truth or human-verified labels. All scored parts are visible; neither real clip tests full disappearance/reappearance. Potential SAM2/DAVIS training overlap prevents interpreting these clips as unseen-data validation. The demonstration supports a runnable named-part recoloring pipeline, not arbitrary faces, 3-D viewpoint consistency, object replacement, or generative video editing.

## Specialist comparison limits

SAM2's dense-mask score and the common patch-localization readout answer different questions. Dense evaluation uses each frame and every nonempty target; patch evaluation uses two-frame tubelets and excludes thin ambiguous visibility. Their absence denominators and rates must remain separate. SAM2 receives quality-100 4:4:4 JPEG conversions while V-JEPA receives the original rendered RGB; conversion error is recorded in each clip manifest. Model objectives, temporal processing, and computational budgets are not matched. V-JEPA can attend to future frames inside each independently encoded 16-frame window, so it is an offline test.
