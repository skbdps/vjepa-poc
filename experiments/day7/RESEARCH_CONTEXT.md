# JEPA-centered persistent-part prediction: research context and review checks

Prepared 2026-10-09. This document explains the candidate and its relationship to primary sources. The separately frozen protocol defines the actual architecture, seeds, training budget and acceptance criteria. Design statements below are hypotheses to test, not completed results.

## Primary-source context

1. **Meta FAIR, official V-JEPA repository.** [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2), especially “V-JEPA 2.1 Pre-training,” the checkpoint table and frozen-probe evaluation instructions. The repository supplies V-JEPA 2.1 models, including ViT-B/16 and ViT-L/16 at 384-pixel resolution, and describes dense predictive loss and intermediate-layer supervision. It is the source of the backbone implementation and checkpoints, not evidence that our particular part tracker works. Pin the exact code revision and checkpoint hash in the experiment.

2. **Mur-Labadia et al., “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.”** [arXiv:2603.14482v2](https://arxiv.org/html/2603.14482v2), particularly Sections 2.2–2.3, the video-object tracking evaluation and Appendix C.3. The paper extends predictive supervision to visible and masked tokens and multiple encoder levels to improve dense representations. Its dense-vision results motivate testing a frozen V-JEPA backbone for tracking. They do not imply that token similarity already preserves individual car or part identity through disappearance.

3. **“Causal-JEPA: Learning World Models through Object-Level Latent Interventions.”** [arXiv:2602.11389v1](https://arxiv.org/html/2602.11389v1), particularly Figure 1, Sections 3–4 and Appendix D. It studies prediction using frozen object-centric representations, with masked-history and future-latent objectives. Its primary object-centric encoder uses VideoSAUR built on frozen DINOv2 features. This is relevant precedent for compact object-level predictive state; it is not an implementation or validation of our V-JEPA part tracker. Its causal interpretation depends on additional assumptions and experiments that our candidate does not establish.

The official repository and both paper versions above were inspected on 2026-10-09. No literature-novelty claim follows from this limited review.

## What this candidate adds to the project

The candidate keeps V-JEPA 2.1 as its visual representation and trains a small persistent-part head. Four part states retain frame-zero-annotated identity anchors, exchange owner context within the known two-car grouping, retrieve current patch evidence and learn presence. A detached future-part latent target supplies the predictive objective. The forecast must also participate in later retrieval, making prediction part of the tested tracking mechanism.

The specific question is: **does future-latent supervision improve persistent part identity beyond an otherwise identical head trained only for retrieval and absence?**

This is supervised adaptation on synthetic training annotations combined with a JEPA-style latent prediction objective. The frozen backbone itself is not retrained. It is not a new self-supervised foundation model, automatic part discovery, an independently established causal model, or a proof of generative video consistency. SAM2 is not the tracker, label generator or latent teacher in this candidate. Earlier SAM2 comparisons remain contextual evidence about the difficulty of the task.

At evaluation, only the initial part annotations may initialize the tracker. A known slot-to-owner grouping is metadata, not a later parent-mask input. Later part and owner masks belong exclusively to scoring. Improvements within this renderer family would justify a further test; they would not establish performance on faces, arbitrary footage or large 3-D viewpoint changes.

## Architecture review checklist

- [ ] The predictive and retrieval-only arms have the same architecture, parameter shapes, initialization, data order and optimization budget for each paired training seed. The intended difference is the weight of the detached predictive loss.
- [ ] The forecast is used by the inference path. With horizon eight tubelets, the forecast issued at time `t` is consumed for time `t+8`, rather than accidentally used as a one-step forecast or left as an unused auxiliary output.
- [ ] Frame-zero-annotated identity anchors remain immutable. Recurrent appearance state may change without overwriting the anchor.
- [ ] Part and owner updates share parameters across arbitrary instance IDs. Part settings remain attached to explicit stable IDs; swapping the input car groups does not introduce a privileged learned “car A” identity.
- [ ] Owner context is derived from the permitted child states and grouping. No extra whole-parent mask is silently supplied at evaluation.
- [ ] Future targets are frozen encoder representations, normalized or transformed by a fixed training-only projection. A detached but jointly trainable target projection must not become a moving or collapsing target.
- [ ] Feature projection is evaluated against a full-dimensional frozen matching baseline so lost backbone information is not mistaken for failure of JEPA prediction.
- [ ] Ambiguous/sliver observations are distinguished from complete absence. The validity rule for each predictive target is explicit; absent parts are not assigned arbitrary zero-vector appearance targets.

## Information-isolation review checklist

- [ ] After frame-zero initialization, synthetic training masks supply losses only. Later ground-truth locations, visibility flags and pooled target latents never enter the recurrent inference state or update gates, including during training rollout.
- [ ] The inference function accepts features, initial annotations and declared metadata only. Ground-truth-bearing objects remain outside that interface.
- [ ] Features come from disjoint 16-frame encoder windows. Every horizon-eight target belongs to the next window, so its source state contains no representation encoded from that target window.
- [ ] A prefix-isolation check changes future-window features and confirms that earlier states and already-issued forecasts remain unchanged.
- [ ] Tracking is described as **offline within each 16-frame encoder window**. Tubelet-zero features can see the first window, so a frame-zero-annotated anchor is not a claim of frame-zero-only visual input.
- [ ] Feature normalization is token-wise or uses fixed training-set statistics. No whole-clip normalization, bidirectional head operation or test-fitted projection leaks future information into a purported forecast.
- [ ] RGB and labels are transformed together before encoder extraction. Flipping cached token coordinates alone is not equivalent to encoding a flipped video.
- [ ] Training, development and held-out scenes have separate recorded seeds. Existing Day4–Day6 observations are prior/development evidence. Test outcomes do not select hyperparameters, thresholds, training seeds or checkpoints.
- [ ] Deterministic scene motion is treated as a shortcut risk. Flips and ID permutations help, but a trained coordinate-only control is needed to assess how much performance follows predictable trajectories without visual identity evidence.

## Measurement and interpretation review checklist

- [ ] Compare predictive versus retrieval-only heads directly, with paired training seeds. A comparison against frozen matching alone cannot isolate the predictive objective.
- [ ] Report all three training seeds. Three models evaluated on the same eighteen videos are not fifty-four independent videos; clip resampling must preserve the paired model/seed structure. Training-seed variation is reported separately or explicitly included in a hierarchical procedure.
- [ ] Report visible localization, wrong-car assignments, absence errors and recovery with their actual denominators. Keep patch/tubelet scores distinct from dense pixel-mask scores.
- [ ] Examine crossing, occlusion, encoder-window boundaries and thin-visibility failures separately. Aggregate improvement can conceal a damaging subgroup regression.
- [ ] Measure future-latent error against simple copy-anchor or copy-last-reference alternatives. Lower latent error alone does not demonstrate better identity tracking.
- [ ] Record training curves, parameter counts, feature-cache provenance and measured runtime. Frozen feature extraction and head training have different costs.
- [ ] A lower auxiliary loss with no tracking improvement is a negative result for the proposed predictive addition.
- [ ] If both trained heads improve over frozen matching but the predictive head does not improve over retrieval-only training, conclude that trainable retrieval helps while the additional JEPA-style objective remains unproven.
- [ ] If a coordinate-only control performs similarly, conclude that the renderer permits a motion shortcut. Do not present that outcome as visual instance understanding.
- [ ] Retain failed seeds and unsuccessful examples. Any revised mechanism needs a new declared development/test boundary.

The intended milestone is a measured benefit from persistent predictive latent state for part identity. The user-facing control registry and recoloring demonstration remain useful downstream artifacts, but they do not substitute for that measurement.
