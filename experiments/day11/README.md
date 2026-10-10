# Direct JEPA → Wan conditioning translator

The translator has been trained and executed through the real Wan generator
on a Colab T4. See `RESULTS.md` for measured progress and remaining failures,
`METHOD.md` for the math, and `PROTOCOL.md` for the fixed first attempt.

This pilot learns a spatial mapping from dense native-image V-JEPA2.1 features
to the actual normalized VAE latents consumed by Wan2.1-VACE-1.3B. It keeps the
pretrained JEPA encoder and generator frozen. The generator receives the
translated latent directly through its trained VACE conditioning branch.
ControlNet supplies the architectural motivation, not pretrained JEPA weights.

Inputs are [1024,24,24] JEPA grids. Outputs are [16,48,48] normalized Wan latents.
The prototype is single-image first; video time alignment remains a separate
test. No masks, displacement, source/target RGB or scene metadata enter F.
The residual inference arm additionally uses the legitimate source VAE latent.

## Files

- `data.py`: exact manifest selection and RGB/hash verification of Day10 pairs.
- `model.py`: 218,112-parameter CNN and 65,728-parameter linear comparator.
- `vae_targets.py`: real, frozen Wan VAE target cache with model/file hashes.
- `train.py`: fixed genuine-image training and complete final-epoch records.
- `vace_bridge.py`: direct conditioning, official mask preparation, and a narrow
  cached-prompt workaround for Diffusers0.35.1's conflicting public validation.
- `evaluate.py`, `render.py`: the original twelve-arm latent/VAE and real VACE
  evaluations. Original implementations remain unchanged by follow-ups.
- `transport.py`, `render_transport.py`: source-appearance transport and its
  complete 27-arm development comparison, including a no-F local-fill control.
- `qualify_development.py`: fixed engineering checks for the declared primary
  method; a passing gate is required before opening new scenes.
- `prepare_fresh.py`, `evaluate_fresh.py`, `render_fresh.py`: separately guarded
  eight-scene evaluation with frozen code, checkpoints, prompt, noise and arms.
- `run_2026-10-10/`: weights, learning curves, manifests, measured outputs,
  independent audits, runtime evidence and readable figures.

## Runtime

Use an existing compatible PyTorch installation; install the small Python
dependencies in `requirements.txt`. T4 renderer execution uses native PyTorch
SDPA, FP16 denoiser and FP32 VAE. Encode text separately and unload its large
encoder before loading the renderer. The first attempt reuses local genuine
JEPA caches and trains the small mapping on CPU while Colab prepares rendering.
No generation API key is required for these public model weights.

```bash
python experiments/day11/vae_targets.py --cache-root DAY10_NATIVE_CACHE --out DAY11_TARGETS
python experiments/day11/train.py make-freeze --jepa-root DAY10_NATIVE_CACHE --targets-root DAY11_TARGETS --out TRAINING_FREEZE.json
```

Publish the exact freeze and sources, verify its returned bytes, and record the
publication receipt required by `train.py` before optimization. Then:

```bash
python experiments/day11/train.py train --jepa-root DAY10_NATIVE_CACHE --targets-root DAY11_TARGETS --out DAY11_TRAINING --freeze TRAINING_FREEZE.json --publication PUBLICATION.json
python experiments/day11/vace_bridge.py
```

The last command runs interface unit checks, not model inference. Real renderer
tests first compare ordinary RGB conditioning with equivalent direct true
VAE latents under identical prompt/noise/settings. The saved actual runs passed
those checks exactly; complete output archives retain every declared arm.

## Running the saved development renderer

The small published prompt cache avoids loading the large text encoder again.
It is bound to the fixed prompt by the saved hashes. After extracting
`run_2026-10-10/runtime/colab_preflight_and_prompt_cache.zip` into `RUNTIME`, use:

```bash
python experiments/day11/render.py --bundle-root experiments/day11/run_2026-10-10/eval_cnn --prompt-cache RUNTIME/prompt_embeddings.pt --prompt-metadata RUNTIME/prompt_metadata.json --out RENDER_CNN --transformer-dtype float16
python experiments/day11/render_transport.py --bundle-root experiments/day11/run_2026-10-10/transport --prompt-cache RUNTIME/prompt_embeddings.pt --prompt-metadata RUNTIME/prompt_metadata.json --out RENDER_TRANSPORT --transformer-dtype float16
```

These commands resolve the exact model revision automatically. A cached
`--snapshot-dir` can avoid repeated downloads. Run GPU jobs sequentially. The
initial renderer performs 32 calls; the transport follow-up performs 62.
Output directories are separate, and resume requires identical bindings.

For unseen-scene evaluation, first bind the actual development gate and both
trained checkpoints in a pretest freeze. Its `evaluation` and `rendering`
objects must exactly match the helper functions in the two fresh runners.
Publish and byte-verify this freeze before `prepare_fresh.py` generates scenes.
The fresh evaluator retains 27 arms for each of CNN and linear on all eight
scenes. The fresh GPU renderer uses the declared nine CNN arms on every scene,
with 104 calls including all ordinary/direct parity checks. It accepts no arm
or scene-selection flag. A repeated run after changing the method is development
work and requires a new untouched holdout for a fresh-test claim.

## Code review findings addressed before training

The original cached-prompt interface was invalid in Diffusers0.35.1: its call
requires a string prompt, while input validation rejects string plus embedding
arguments. A scoped `encode_prompt` replacement now supplies cached tensors
while preserving the upstream call and validation. Both overrides restore the
original methods even after an exception.

GroupNorm uses statistics over spatial positions. Consequently the CNN can
change predictions outside an edited patch even though JEPA tokens there are
unchanged. Regional metrics are required. Exact zero-edit conditioning identity
does not establish exact pixel preservation after diffusion. A CNN/linear
comparison does not establish JEPA's advantage over other control signals.

## Primary implementation references

- [Original ControlNet code](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py)
- [Pinned Wan VACE pipeline](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/wan/pipeline_wan_vace.py)
- [Pinned Wan VACE transformer](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/transformer_wan_vace.py)
- [Exact renderer weight revision](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers/tree/ec4d2cb062b548996b179d493fdd05340de702a1)

