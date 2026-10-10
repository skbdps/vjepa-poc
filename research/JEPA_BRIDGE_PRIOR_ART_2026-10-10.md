# Existing bridges for JEPA editing: prior art and reuse decision

Reviewed 2026-10-10. Research-only update; no model download, installation, training, inference, Colab execution, or experimental result is reported here.

## Decision

Do not begin by assuming that a new small adapter will solve the interface. The user has identified ControlNet as the intended reference. Assess a ControlNet-style spatial conditioning branch for the direct JEPA-to-generator connection. Separately, VideoRAE's released decoder is a candidate for testing whether our edits can be reconstructed before attempting a larger generator integration. The published JEPA-to-Cosmos bridge remains a semantic-conditioning comparison; Crosscoders is an alternative research direction.

The initial search could not recover the remembered name. During this review the user explicitly confirmed it was ControlNet. Crosscoders and CycleGAN below are comparison methods, not the user's recalled reference.

## ControlNet: the confirmed reference

The [original paper](https://arxiv.org/abs/2302.05543) supplies an established spatial-conditioning architecture: preserve a pretrained diffusion backbone and learn an additional branch connected through zero-initialized convolutions. Its released controls include edges, depth and poses. This is a concrete mechanism to build on, not a pretrained interpreter of arbitrary JEPA features.

Our proposed direct application is to adapt the conditioning input to dense edited JEPA features with explicit destination coordinates and retain source appearance through a separate compatible reference path. The branch learns how to affect denoising; it need not reconstruct the entire generator latent from JEPA alone. This is a design hypothesis, not an implemented result.

Two distinct tests must stay separate:

- Direct JEPA conditioning: train/adapt the control branch for the actual feature tensor, spatial grid and encoder. Supervise first on genuine target encodings; then evaluate edited states against independent target renders. Include fixed-caption tests so language cannot carry the edit instead.
- Decoded-guide baseline: convert our JEPA state into supported layout/depth/edge guides and use a pretrained controller. This tests an indirect interface and requires an equal-guide non-JEPA baseline.

Frozen generator weights do not imply unchanged output pixels. Source identity, unaffected regions and temporal consistency remain separate requirements. Original ControlNet is an image architecture; an image pilot can isolate the interface, but frame-independent success cannot establish video consistency.

The [official training guide](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) illustrates learning custom controls and discusses memory constraints. Neither that guide nor our prior T4 runs benchmarks a dense JEPA-conditioned video branch.

## Comparison with other alignment approaches

| Approach | Established function | Implication for this project |
|---|---|---|
| Crosscoders | Joint sparse feature dictionaries across model activations; newer work transfers steering directions across language models. | Promising for corresponding changes. Ordinary joint encoding needs activations from both models and is not an automatic JEPA-only decoder. |
| CycleGAN | Unpaired image-domain translation with adversarial and round-trip objectives. | Cycle consistency can regularize a mapping but cannot alone establish the intended spatial correspondence or edit meaning. |
| ControlNet | A trained conditional branch adds spatial controls to a frozen diffusion model. | Reuse an existing supported guide, or train a new feature interface. Raw JEPA vectors cannot simply replace a depth/edge input. |

Sources and available implementations:

- [Cross-Architecture Model Diffing with Crosscoders, section 3.1](https://arxiv.org/html/2602.11729v1): tests transfer of steering vectors through corresponding shared dictionary directions. This is empirical LLM evidence, not JEPA/video validation.
- [Neel Nanda's Crosscoders](https://github.com/neelnanda-io/Crosscoders): MIT implementation.
- [Anthropic's DFC explanation](https://www.anthropic.com/research/diff-tool): separates shared and model-exclusive features.
- [CycleGAN/pix2pix official code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- [ControlNet official code](https://github.com/lllyasviel/ControlNet).

A further relevant baseline is [Latent Space Translation via Semantic Alignment](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract.html): algebraic mappings between pretrained representation spaces. The linked [Latentis library](https://github.com/Flegyas/Latentis) is Apache-2.0 and marked under construction. Its existence does not establish that JEPA and a renderer differ only by a simple change of coordinates.

## Existing JEPA rendering connections

### VideoRAE: first compatibility-audit candidate

[Project](https://zhxie0117.github.io/VideoRAE/), [code](https://github.com/zhxie0117/VideoRAE), [weights](https://huggingface.co/siuuuuuuxzh/videorae/tree/main).

Public files include vjepa21_3d.pth, vjepa_rae.pth and vjepa_vqrae.pth. These support inspecting a pretrained reconstruction route before training a translator. Reconstruction capability is not evidence of arbitrary edited-state fidelity or compatibility with an unrelated frozen video generator.

The [3D implementation](https://github.com/zhxie0117/VideoRAE/blob/main/models/model_sem/auto_vjepa21_3d.py) uses a V-JEPA 2.1 ViT-B teacher, default multiscale layers [2,5,8,11], 768-dimensional tokens and a structured 8x16x16 teacher grid for 16 frames at 256x256. Our ViT-L/384 features and native single-image features cannot be substituted by reshaping. Checkpoint-specific configuration remains to be confirmed. Re-encoding with the required teacher would require revalidating our edit operation.

Source blob inspected: fbcbe4d67abd4b10a6cfa23b3be4f2faf9224dd5.

### JEPA Guided Diffusion: released bridge, important spatial limitation

[Code](https://github.com/AlterraFa/JEPA-Guided-Diffusion), [published checkpoint](https://huggingface.co/AlterraLaniakea/jepa-guided-diffusion/tree/main), [paper](https://arxiv.org/html/2609.21379v1).

It maps V-JEPA features plus captions through a QFormer/Llama predictor into Cosmos conditioning, while retaining source RGB frames. Predictor training uses cached features and caption-derived targets. The code is MIT; model/dependency terms are separate. This supplies an actual reuse baseline, not a universal latent converter.

The [published configuration](https://huggingface.co/AlterraLaniakea/jepa-guided-diffusion/blob/main/config.yaml) selects QFormer. The [predictor implementation](https://github.com/AlterraFa/JEPA-Guided-Diffusion/blob/main/jepa_guidance/modeling/predictor.py) performs row-wise visual projection and normalization followed by query-to-visual cross-attention, without adding new destination coordinates to the visual rows.

**Code-derived inference, not an executed benchmark:** with caption fixed and an all-valid visual mask, in evaluation mode a pure permutation P of visual rows leaves the bridge output invariant, apart from numerical roundoff:

B(PZ, c) = B(Z, c).

Permuting keys and values together leaves attention's weighted sum unchanged. Later processing of the same summary cannot recover the lost row assignment. Position information embedded within an original feature vector moves with that vector; it does not identify its new destination. Copy-and-repair changes the token multiset as well, so that complete edit is not necessarily invisible. Nevertheless, explicit destination control is missing from this route.

Source blob inspected: ec2000d6d22309e80d72431462ab424d022d6f43.

### Relevant designs without verified usable releases

[PHANTOM](https://plan-lab.github.io/projects/phantom/) couples visual and physical branches and demonstrates force-conditioned generation after further training. Its official page currently says code is coming soon. It should not be described as a currently available drop-in bridge. [RoboJEPA's project page](https://robojepa.github.io/) also says coming soon; a usable decoder release was not verified.

## Proposed direction-transfer alternative

Our hypothesis, not an existing validated capability:

h_G_edited = h_G_source + T(layer, noise_level, spatial_context, Z_J_edited - Z_J_source).

This asks the bridge to communicate the requested change while preserving the generator's appearance-bearing source state. Crosscoder steering transfer provides precedent for the general strategy. JEPA-to-video transfer still needs paired feature correspondence, a chosen generator layer/noise level, destination coordinates, and tests of actual rendered behavior.

A good activation fit is insufficient. In Day10, lower regional JEPA error did not always improve RGB hole error. Protected content must also be measured through the new decoder.

## Next bounded test, once execution is authorized

1. Verify a released decoder's checkpoint, teacher, preprocessing and genuine reconstruction on unseen scenes. Profile memory before claiming T4 feasibility.
2. Compare genuine target encodings with our edited encodings under exactly that teacher. Genuine targets are privileged diagnostics, not deployable controls.
3. Change only the JEPA input while fixing source reference, prompts and other controls. Include no-edit and wrong-direction controls.
4. Measure requested displacement, source identity/texture, protected content and temporal quality together, using rendered pixels and independent measurements rather than only the existing pointwise JEPA probe.
5. If genuine reconstruction fails, diagnose compatibility/decoder issues. If genuine targets work but edited states fail, examine the edit and distribution shift. If edited states work, then test whether an external generator preserves their constraints across seeds and revisions.

No current paper, checkpoint listing, or small parameter count establishes the memory, data or training budget for our full proposed system.
