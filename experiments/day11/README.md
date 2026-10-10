# Direct JEPA → Wan conditioning translator

Implementation in progress; do not interpret the existence of code as a passed
rendered-edit experiment. See `PROTOCOL.md` for the fixed first attempt.

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
tests must first compare ordinary RGB conditioning with equivalent direct true
VAE latents under identical prompt/noise/settings. Actual results will be stored
separately with failures retained.

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

