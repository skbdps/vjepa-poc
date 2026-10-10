# Day11: direct JEPA to Wan VACE conditioning translator

Status: initial protocol, written before bridge training or rendered evaluation.

## Question

Can a small spatial translator map dense V-JEPA2.1 features to the normalized
VAE conditioning latents consumed by an existing video generator, and preserve
a prescribed local JEPA edit through that interface?

This first stage uses native single images to isolate translation. A successful
image bridge is not evidence of temporal consistency, realistic physics,
face identity, or generalization to real footage. ControlNet motivates the
frozen-generator conditioning route; the actual renderer is Wan2.1-VACE-1.3B.
We do not train an original ControlNet or replace JEPA with segmentation.

## Exact interfaces

- Frozen JEPA: official V-JEPA2.1 ViT-L/384, revision and weight digest recorded
  by Day10. Native image input [B,3,1,384,384], features [B,1024,24,24].
- Translator F: train-only feature standardization, dense spatial projection,
  explicit destination coordinates, small convolutional blocks, 2x spatial
  upsampling to [B,16,48,48]. Keep spatial order; no unordered query pooling.
- Frozen Wan VAE: AutoencoderKLWan from
  `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, resolved revision recorded at execution.
  Image input [B,3,1,384,384] in [-1,1]. Use posterior mode, not a random sample.
  Target A(x) = (mode(encode(x)) - config.latents_mean) / config.latents_std.
- Frozen VACE branch: inject predicted A(x) directly at its conditioning-latent
  interface using Diffusers 0.35.1. For all-known video conditioning, concatenate
  inactive A(x), reactive A(zeros in normalized RGB space), and the official
  packed zero mask. The reactive component must not be replaced by literal
  latent zeros. No decode/re-encode between F and VACE.
- Denoiser, text encoder and VAE weights stay frozen. The translator trains
  against cached true latents, without full-generator backpropagation.

## Data and initial training

Use only the Day10 native-image manifest: 32 train scenes (13400–13431, two
genuine views each) and eight development scenes (13500–13507, two views).
Recreate RGB frame15 from the fixed renderer and verify exact stored RGB hashes.
Read manifest entries, not directory globs, to exclude incomplete staging files.
Store VAE targets separately with source hashes, normalization and model revision.

Initial fixed run: seed1111, AdamW, learning rate0.001, weight decay0.0001,
batch4, 150epochs, FP32, width64. Train genuine-image latent MSE only. Compare
a linear spatial projection baseline under the same data and optimizer budget.
Retain final checkpoints and all learning curves, including failures. Use no
target mask, displacement, target RGB or metadata as translator inputs.
Development may guide explicitly documented follow-ups; it is not a final test.
Any changed budget, model, loss or normalization is a separately named attempt.

## Diagnostics and interventions

Evaluate two predeclared output routes:

1. Absolute: F(z_edit).
2. Source-preserving residual: A(x_source) + F(z_edit) - F(z_source).

The second uses source appearance legitimately and enforces exact no-edit at
the conditioning-latent level. It does not guarantee exact output pixels.

For a fixed source, prompt, noise seed and scheduler, compare: source/no edit,
genuine target JEPA (privileged interface diagnostic), source-only copy/repair
JEPA, wrong-direction JEPA, and spatially shuffled JEPA. Also use true source
and true target Wan latents as renderer-interface diagnostics. Never report an
oracle input as a deployable editing result. Keep prompts identical and free
of edit direction. Verify exact interface equivalence before relying on output.

Measure normalized latent MSE and decoded/rendered RGB MSE globally and in
source hole, destination, distractor and protected background; object centroid
and color/appearance where the fixed synthetic masks permit evaluation. Report
images, failures, and per-scene results alongside averages. Latent movement
alone is insufficient. A VAE reconstruction establishes the decoder ceiling.

## Fresh evaluation and stopping interpretation

Existing Day10 tests13600–13603 are historical development examples if reused;
they cannot become a new untouched test. Before new evaluation, freeze code,
checkpoint hashes and settings. Reserve eight fresh scenes14000–14007 with
offsets +32,-32,+48,-48,+64,-64,+80,-80. Encode source and independently rendered
target natively in FP32. No adaptation after their first inspection unless
those scenes are explicitly redesignated development and a new holdout is used.

If genuine target JEPA cannot translate, diagnose representation/bridge capacity
before interpreting edited features. If true Wan target control does not guide
the renderer, fix renderer configuration before judging F. If genuine target
JEPA works but edited JEPA fails, investigate edit distribution and lost detail.
If all signals work in the image pilot, temporal alignment is the next distinct
experiment; repeating native-image VAE latents is not a validated video bridge.

Publish executable code, actual runtime, model revisions, all attempts, evidence
and limitations to the existing research branch. Do not claim the broad creative
control problem is solved from this bounded pilot.
