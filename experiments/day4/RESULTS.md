# Persistent parts: experiment result

This experiment produced a usable part-controlled video prototype: initialize a named door and window once, propagate their masks, then change either part's color throughout the clip. The saved mask registry keeps the same IDs, and the offline editor reuses them without model inference. This is deterministic compositing on tracked masks; it does not generate new video or solve face identity.

The editor keeps independent settings for each part, supports simultaneous edits, and exports/imports a validated JSON recipe. A native replay command applies that recipe to the saved masks. The committed two-part previews color the door blue and window amber on both real clips; all 104 frames pass the prediction-relative outside-region and overlap-preservation checks.

The most useful evidence is the combination of a real editing demonstration and a harder held-out tracking test. The results do not establish that frozen V-JEPA is the best component for this task.

## Real video evidence

Two DAVIS sequences were initialized with approximate assistant-authored polygons on frame zero. Later polygons were authored from the source frames before viewing predictions and supplied only to evaluation. They are sparse diagnostic labels, not official DAVIS part annotations or human-verified ground truth.

| Sequence | Frames | Scored part-frames | SAM2.1 Tiny mean IoU | Per-part affine flow | Shared-plane homography |
| --- | ---: | ---: | ---: | ---: | ---: |
| car-roundabout | 64 | 16 | 86.18% | 42.01% | 74.64% |
| car-shadow | 40 | 10 | 81.48% | 64.52% | 52.10% |

On car-roundabout, SAM2's door IoU is 91.94% and window IoU is 80.41%. The door is the stronger demonstration. The weak per-part flow baseline loses the window, so the comparison also includes shared-plane tracking that borrows texture from the larger car region. None of the three trackers was retuned on car-shadow.

The edit function changes only the selected predicted mask after subtracting every unselected predicted mask. Array checks verify zero changed pixels outside that permitted region before lossy video encoding. This guarantee is relative to the predictions: mask spill can still alter the wrong physical surface. Against the approximate roundabout polygons, the protected door mask has 93.62% mean precision and 98.14% mean recall.

These clips contain viewpoint, scale, and illumination changes, but no complete disappearance of the target parts. They demonstrate persistent alignment and selective control, not real-world identity recovery after occlusion. DAVIS may overlap SAM2's training data, so these are pipeline demonstrations rather than evidence of unseen-data generalization.

## Held-out synthetic V-JEPA result

The Day 4 benchmark has 64-frame clips, four tagged parts on two cars, and three conditions: prolonged occlusion, crossing, and scale/camera changes. Six calibration clips set presence thresholds, six separate development clips select the configuration, and eighteen untouched test clips measure it. The selected configuration and source digest were committed before the test ran.

The selected configuration was `no_memory`: fixed initial appearance with contextual features and motion guidance. The selection objective averages visible localization and absent-part specificity. The full appearance-update configuration did not win development selection.

| Method | Correct visible localizations | False presence while absent | Recovery events |
| --- | ---: | ---: | ---: |
| Global V-JEPA matching | 1,064 / 1,922 (55.36%) | 53 / 260 (20.38%) | 7 / 18 |
| **Selected V-JEPA tracker (`no_memory`)** | **1,026 / 1,922 (53.38%)** | **14 / 260 (5.38%)** | **5 / 18** |
| Full appearance-update tracker | 1,082 / 1,922 (56.30%) | 29 / 260 (11.15%) | 5 / 18 |
| Multiscale template + flow | 1,215 / 1,922 (63.22%) | 63 / 260 (24.23%) | 0 / 18 |

The selected tracker reduces hidden-part false presence substantially but loses visible localization accuracy. It does not beat the classical comparator on visible localization. Repeated same-part/different-car confusion remains a concrete obstacle to persistent identity. Fifty ambiguous target-tubelets are excluded by the declared scoring rule, equally for every method.

Paired resampling of the eighteen clips puts the selected tracker's visible-localization difference versus the template at −9.83 percentage points (95% interval −16.03 to −3.71), and its hidden false-presence difference at −18.85 points (−27.48 to −3.67). Lower false presence is better. Its balanced-objective advantage is +4.51 points, with an interval from −3.53 to +9.85; this does not establish a clear overall winner.

The crossing condition is the sharpest failure: selected V-JEPA localizes 271/702 visible samples correctly (38.60%) versus the template's 430/702 (61.25%), and assigns 173/702 to the wrong car. On long occlusion, visible localization is nearly tied, while the selected tracker's hidden false-presence rate is much lower. [The independent audit](AUDIT.md) preserves the exact condition counts and checks.

A supplementary recovery-delay analysis prevents an overstatement: zero immediate template recoveries does not mean it never recovers. Within four source frames of first scored reappearance, selected V-JEPA has 12/18 successes and the template 2/18. At an eight-frame horizon, the template has 16 known successes versus 12 for selected V-JEPA, but only six event horizons are fully observed before clip end; one V-JEPA outcome remains unresolved. This analysis was added after the V-JEPA results, is explicitly post-hoc, and does not replace the frozen primary metric.

V-JEPA uses offline attention inside each independent 16-frame window. The synthetic cars are 2-D sprites, and their first four frames are stationary. Frames are correlated; uncertainty must be resampled by clip, not by individual frame.

## Specialist comparison

The completed SAM2.1 Tiny test uses the same eighteen held-out clips and frame-zero part annotations. Its mask-to-patch readout and calibration threshold were frozen before testing. The common patch task has 1,922 visible and 260 fully absent target-tubelets, with fifty ambiguous cases excluded for every method.

| Common patch metric | SAM2.1 Tiny | Selected V-JEPA (`no_memory`) | Template + flow |
| --- | ---: | ---: | ---: |
| Correct visible localization | 1,826/1,922 (95.01%) | 1,026/1,922 (53.38%) | 1,215/1,922 (63.22%) |
| False presence while absent | 11/260 (4.23%) | 14/260 (5.38%) | 63/260 (24.23%) |
| First-reappearance localization | 6/18 (33.33%) | 5/18 (27.78%) | 0/18 (0%) |
| Wrong car on visible targets | 55/1,922 (2.86%) | 183/1,922 (9.52%) | 26/1,922 (1.35%) |
| Defined car-identity transitions | 4 | 60 | 18 |

SAM2 minus selected V-JEPA gives the following differences, using 2,000 paired whole-clip bootstrap draws and seed 914. Values are percentage points; a negative hidden false-presence difference favors SAM2.

| Metric | Difference | Descriptive 95% interval |
| --- | ---: | ---: |
| Visible localization | +41.62 | [+30.94, +50.10] |
| Hidden false presence | −1.15 | [−5.80, +3.41] |
| First-reappearance localization | +5.56 | [−12.50, +27.78] |

SAM2 substantially improves visible localization in this renderer family. The other two intervals include zero, so the test does not establish a hidden-presence or immediate-recovery advantage over selected V-JEPA. Against template + flow, SAM2 improves visible localization by +31.79 points [+22.60, +40.09]. Its crossing failures remain material: all 55 wrong-car checks occur there, and only 2/6 crossing events recover immediately. A sustained wrong identity can generate many wrong-car checks after a single identity transition. Supplementary recovery reaches 8/18 by two source frames and remains 8/18 at four; at eight frames, only six horizons are complete (2/6 recovered), with eight known successes overall and six unresolved censored events.

The separate raw-mask readout gives **94.42% mean IoU over 3,997 visible part-frames**, 94.22% pooled visible-pixel precision, and 95.10% pooled visible-pixel recall. It reports nonempty masks on **60/539 absent part-frames (11.13%)**, totaling 75,016 predicted pixels while the target is absent. Dense scoring includes every frame after frame zero and any nonempty target, including thin slivers. It applies no patch-presence threshold. The 11.13% dense false-presence rate and 4.23% patch false-presence rate therefore have different units, denominators, and readouts; they are not interchangeable.

A supplementary mask-presence gate copies the original calibration threshold unchanged and retains a mask in both frames only when that pair passes the frozen patch readout. Its exact policy was frozen during test execution before held-out scores were viewed; it neither changes the primary results nor introduces a fitted parameter. Hidden mask reports fall from **60/539 to 30/539 (11.13% to 5.57%)**, but visible mean IoU falls from **94.42% to 92.20%** and visible pixel recall from 95.10% to 94.43%. It removes 98 nonempty masks on visible targets and 30 on absent targets. In the 81-part-frame diagnostic for visible parts without a qualifying ground-truth patch in their pair, it removes 65 masks and collapses mean IoU from **73.91% to 0.011%**. This is a severe loss on thin, partially visible, or moving parts, not merely a small overall accuracy cost. **Raw masks remain the default**; this gate is an explicit tradeoff, not a general improvement. The score measures patch coverage rather than calibrated probability, and the two-frame decision is not causal streaming.

See the [full frozen comparison](run_2026-10-09/comparison/comparison.md), [machine-readable counts and intervals](run_2026-10-09/comparison/comparison.json), [gate tradeoff](run_2026-10-09/presence_gate/test/presence_gate_results.md), and [post-test failure gallery](run_2026-10-09/failures/README.md). The gallery deliberately selects failures after scoring; it is not a representative sample. [The audit](AUDIT.md) independently reproduces the counts and paired intervals and checks the frozen source, threshold, manifests, and saved mask hashes.

SAM2 is a specialist mask tracker with sequential memory, while V-JEPA supplies frozen patch features with offline attention inside separate windows. Training objectives, temporal processing, resolution, and compute are not matched. SAM2 receives measured quality-100 JPEG conversions and V-JEPA receives original rendered RGB. These small-sample synthetic intervals do not establish real-video generalization.

## What this supports

The practical architecture is a persistent object/part registry, a video mask tracker, and a separately controlled edit operation. SAM2 supplies the strongest visible-localization result in this test and usable masks for the real-video recoloring prototype. Mask tracking and edit rendering can be reused independently. The current demonstration supports simple localized recoloring after manual initialization; raw masks remain the default because the supplementary gate destroys much of the thin-visibility signal.

The next research gate is unambiguous object identity through full occlusion and similar-object crossings on newly captured, independently annotated footage. Neither high average mask IoU nor stable part IDs resolves the observed identity and recovery failures. Automatic part discovery, faces, large viewpoint changes, replacement of objects, and generative temporal consistency remain untested. This run does not establish a benefit from online V-JEPA appearance memory or a necessary role for frozen V-JEPA in the editing pipeline; any proposed role needs a measured improvement over the specialist and classical baselines.
