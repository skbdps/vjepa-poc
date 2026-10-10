"""Direct, normalized latent conditioning for the frozen Wan VACE renderer.

Pinned interface: diffusers==0.35.1, Wan-AI/Wan2.1-VACE-1.3B-diffusers.
This module does not train a mapper or claim that conditioning guarantees edits.
It supplies the mapper's output to VACE without rendering it to RGB first.

For F=4k+1 frames at H x W (H,W divisible by 16), VACE consumes:
    [N(E(video * (1-mask))), N(E(video * mask)), packed_mask]
    [1, 16 + 16 + 64, (F-1)//4+1, H//8, W//8].
N(z) = (z - vae.config.latents_mean) / vae.config.latents_std.
Mask 0 is known conditioning; mask 1 is a generation/control region. Neither
constitutes a hard pixel lock. For all-known video, the reactive half is E(0)
after normalization, NOT a literal zero latent. Zero video here means RGB zero
in [-1,1] preprocessing space (mid-gray), not a black pixel image.

Reference-image slots are different: official code prepends [N(E(ref)), 0],
with zero mask, along the latent-time dimension. This first pilot deliberately
rejects references, rather than silently assigning them ordinary-frame time.

Sources inspected at v0.35.1:
  src/diffusers/pipelines/wan/pipeline_wan_vace.py
    prepare_video_latents, prepare_masks, __call__
  src/diffusers/models/transformers/transformer_wan_vace.py
    WanVACETransformer3DModel.forward(control_hidden_states=...)

The override changes prepare_video_latents and the mask supplied to the
official prepare_masks method. A narrowly scoped encode_prompt override also
works around v0.35.1's conflicting public cached-embedding validation: __call__
requires a string prompt, but check_inputs rejects a string plus embeddings.
Cached tensors are returned by encode_prompt, never passed to that validation.
CFG, control patch embedding, all VACE blocks,
noise initialization, scheduler, decoding and output handling remain upstream.
Use one pipeline serially: temporary overrides are not thread-safe.
"""

from contextlib import contextmanager
from importlib.metadata import version
import inspect
from types import MethodType, SimpleNamespace

import torch


DIFFUSERS_VERSION = "0.35.1"
MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"


def _validate_latent(latent, channels=16):
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        raise ValueError("Expected a [1,C,T,H,W] torch tensor.")
    if latent.shape[0] != 1 or latent.shape[1] != channels:
        raise ValueError(f"Expected batch 1 and {channels} channels; got {tuple(latent.shape)}.")
    if min(latent.shape[2:]) < 1 or not latent.is_floating_point():
        raise ValueError("Latent dimensions must be positive and dtype floating point.")
    if not torch.isfinite(latent).all().item():
        raise ValueError("Latents contain NaN or infinity.")


def _normalization(vae, latent):
    config = vae.config
    if config.z_dim != 16:
        raise ValueError("This interface requires the 16-channel Wan2.1 VAE.")
    mean = torch.as_tensor(config.latents_mean, device=latent.device, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.as_tensor(config.latents_std, device=latent.device, dtype=torch.float32).view(1, 16, 1, 1, 1)
    if not torch.isfinite(mean).all().item() or not (torch.isfinite(std) & (std > 0)).all().item():
        raise ValueError("Invalid VAE normalization statistics.")
    return mean, std


def normalize_wan_latents(vae, raw_latents):
    """Normalize a VAE posterior mode, in float32; do not normalize twice."""
    _validate_latent(raw_latents)
    mean, std = _normalization(vae, raw_latents)
    return (raw_latents.float() - mean) / std


def denormalize_wan_latents(vae, normalized_latents):
    """Inverse for a direct VAE diagnostic decode; not required by the bridge."""
    _validate_latent(normalized_latents)
    mean, std = _normalization(vae, normalized_latents)
    return normalized_latents.float() * std + mean


def _validate_mask(mask, frames, height, width):
    expected = (1, 1, frames, height, width)
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != expected:
        raise ValueError(f"pixel_mask must have shape {expected}.")
    if not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError("pixel_mask must be binary, with 0 known and 1 generation/control.")


@torch.no_grad()
def encode_vace_video_condition(vae, video, pixel_mask):
    """Cache official teacher payload [inactive, reactive], without a generator.

    video: float [1,3,F,H,W] already normalized to [-1,1], on the VAE device.
    Return float32 normalized [1,32,T,H/8,W/8]. Uses posterior mode, not sample.
    This function is for oracle/cache creation; inference translator output
    goes straight to direct_vace_call without this encode/decode step.
    """
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[:2] != (1, 3):
        raise ValueError("video must have shape [1,3,F,H,W].")
    _, _, frames, height, width = video.shape
    if frames % 4 != 1 or height % 16 or width % 16:
        raise ValueError("Require F=4k+1 and H,W divisible by 16.")
    if not video.is_floating_point() or not torch.isfinite(video).all().item():
        raise ValueError("video must be finite and floating point.")
    if video.min().item() < -1 or video.max().item() > 1:
        raise ValueError("video must already be normalized to [-1,1].")
    _validate_mask(pixel_mask, frames, height, width)
    video = video.to(dtype=vae.dtype)
    mask = pixel_mask.to(device=video.device, dtype=vae.dtype)
    halves = []
    for source in (video * (1 - mask), video * mask):
        raw = vae.encode(source).latent_dist.mode()
        halves.append(normalize_wan_latents(vae, raw))
    result = torch.cat(halves, dim=1)
    expected = (1, 32, (frames - 1) // 4 + 1, height // 8, width // 8)
    if tuple(result.shape) != expected:
        raise ValueError(f"Unexpected VAE shape {tuple(result.shape)}, expected {expected}.")
    return result


def assemble_all_known_condition(normalized_image_latent, normalized_empty_video_latent):
    """Use a cached actual N(E(0)); a literal zero tensor is not equivalent."""
    _validate_latent(normalized_image_latent)
    _validate_latent(normalized_empty_video_latent)
    if normalized_image_latent.shape != normalized_empty_video_latent.shape:
        raise ValueError("Image and empty-video latents must have matching shapes.")
    return torch.cat(
        [normalized_image_latent, normalized_empty_video_latent.to(normalized_image_latent)], dim=1
    )


def _check_pipeline(pipe):
    if version("diffusers") != DIFFUSERS_VERSION:
        raise RuntimeError(f"Use diffusers=={DIFFUSERS_VERSION}; an unreviewed interface may change mask/time semantics.")
    from diffusers import WanVACEPipeline

    if not isinstance(pipe, WanVACEPipeline):
        raise TypeError("Expected WanVACEPipeline.")
    config = pipe.transformer.config
    if config.in_channels != 16 or config.vace_in_channels != 96 or tuple(config.patch_size) != (1, 2, 2):
        raise ValueError("Unexpected transformer channel/patch configuration.")
    if pipe.vae_scale_factor_temporal != 4 or pipe.vae_scale_factor_spatial != 8:
        raise ValueError("Unexpected VAE strides.")
    if "control_hidden_states" not in inspect.signature(pipe.transformer.forward).parameters:
        raise ValueError("Transformer has no direct VACE control interface.")


def _validate_prompt_embeddings(embeddings, name):
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise ValueError(f"{name} must be a [1,L,4096] tensor.")
    if embeddings.shape[0] != 1 or embeddings.shape[2] != 4096 or not 1 <= embeddings.shape[1] <= 512:
        raise ValueError(f"{name} must have shape [1,L,4096], with 1 <= L <= 512.")
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all().item():
        raise ValueError(f"{name} must be floating point and finite.")


@contextmanager
def _temporary_cached_prompt(pipe, positive, negative):
    """Return cached T5 outputs at the official encode_prompt boundary."""
    if getattr(pipe, "_day11_cached_prompt_active", False):
        raise RuntimeError("Nested/concurrent cached-prompt calls are unsupported.")
    had_instance_method = "encode_prompt" in pipe.__dict__
    old_instance_method = pipe.__dict__.get("encode_prompt")

    def encode_prompt(self, prompt, negative_prompt=None, do_classifier_free_guidance=True,
                      num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None,
                      max_sequence_length=512, device=None, dtype=None):
        if prompt_embeds is not None or negative_prompt_embeds is not None:
            raise ValueError("Cached tensors must enter through the scoped override only.")
        if num_videos_per_prompt != 1 or max_sequence_length != positive.shape[1]:
            raise ValueError("Cached prompt batch/sequence configuration does not match inference.")
        if do_classifier_free_guidance and negative is None:
            raise ValueError("Classifier-free guidance requires cached negative embeddings.")
        device = device or self._execution_device
        result_positive = positive.detach().to(device=device, dtype=dtype or positive.dtype)
        result_negative = None
        if do_classifier_free_guidance:
            result_negative = negative.detach().to(device=device, dtype=dtype or negative.dtype)
        return result_positive, result_negative

    pipe._day11_cached_prompt_active = True
    pipe.encode_prompt = MethodType(encode_prompt, pipe)
    try:
        yield
    finally:
        if had_instance_method:
            pipe.encode_prompt = old_instance_method
        else:
            delattr(pipe, "encode_prompt")
        delattr(pipe, "_day11_cached_prompt_active")


def _cached_prompt_call(pipe, *, prompt_embeds, negative_prompt_embeds=None, **pipeline_kwargs):
    _validate_prompt_embeddings(prompt_embeds, "prompt_embeds")
    if negative_prompt_embeds is not None:
        _validate_prompt_embeddings(negative_prompt_embeds, "negative_prompt_embeds")
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError("Positive and negative cached embeddings must have matching shapes.")
    if pipeline_kwargs.get("guidance_scale", 5.0) > 1 and negative_prompt_embeds is None:
        raise ValueError("Classifier-free guidance requires cached negative embeddings.")
    sequence_length = prompt_embeds.shape[1]
    if pipeline_kwargs.get("max_sequence_length", sequence_length) != sequence_length:
        raise ValueError("max_sequence_length must match the cached embedding length.")
    pipeline_kwargs["max_sequence_length"] = sequence_length
    # A supplied string records the actual cached text. If absent, an empty
    # placeholder satisfies upstream validation; it is never tokenized.
    prompt = pipeline_kwargs.pop("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string when supplied; cached content is not re-encoded.")
    with _temporary_cached_prompt(pipe, prompt_embeds, negative_prompt_embeds):
        return pipe(prompt=prompt, **pipeline_kwargs)


def cached_prompt_vace_call(pipe, *, prompt_embeds, negative_prompt_embeds=None, **pipeline_kwargs):
    """Official RGB-condition pipeline with cached prompts for parity testing.

    Unlike a raw v0.35.1 pipe(prompt_embeds=...) call, this bypasses its public
    validation conflict while preserving check_inputs and the denoising loop.
    Supply the actual prompt string for readable provenance when available.
    """
    _check_pipeline(pipe)
    return _cached_prompt_call(pipe, prompt_embeds=prompt_embeds,
                               negative_prompt_embeds=negative_prompt_embeds, **pipeline_kwargs)


@contextmanager
def _temporary_direct_condition(pipe, condition, pixel_mask):
    """Private scoped injection. Exceptions restore original bound methods."""
    if getattr(pipe, "_day11_direct_condition_active", False):
        raise RuntimeError("Nested/concurrent direct calls on one pipeline are unsupported.")
    original_prepare_masks = pipe.prepare_masks
    replaced = ("prepare_video_latents", "prepare_masks")
    old_instance_values = {name: pipe.__dict__[name] for name in replaced if name in pipe.__dict__}

    def prepare_video_latents(self, video, mask, reference_images=None, generator=None, device=None):
        if reference_images is not None and any(len(refs) for refs in reference_images):
            raise ValueError("Reference slots are not enabled for this pilot.")
        expected = (1, 3, pixel_mask.shape[2], pixel_mask.shape[3], pixel_mask.shape[4])
        if tuple(video.shape) != expected:
            raise ValueError(f"Upstream condition geometry changed: {tuple(video.shape)} != {expected}.")
        return condition.detach().to(device=device or video.device, dtype=self.vae.dtype)

    def prepare_masks(self, mask, reference_images=None, generator=None):
        # Preserve upstream 8x8 spatial packing and nearest-exact time sampling.
        result = original_prepare_masks(
            pixel_mask.to(device=mask.device, dtype=mask.dtype), reference_images, generator
        )
        expected = (1, 64, *condition.shape[2:])
        if tuple(result.shape) != expected:
            raise ValueError(f"Upstream packed mask shape {tuple(result.shape)} != {expected}.")
        return result

    pipe._day11_direct_condition_active = True
    pipe.prepare_video_latents = MethodType(prepare_video_latents, pipe)
    pipe.prepare_masks = MethodType(prepare_masks, pipe)
    try:
        yield
    finally:
        for name in replaced:
            if name in old_instance_values:
                setattr(pipe, name, old_instance_values[name])
            else:
                delattr(pipe, name)
        delattr(pipe, "_day11_direct_condition_active")


def direct_vace_call(pipe, conditioning_latents, *, pixel_mask, **pipeline_kwargs):
    """Run official inference with learned normalized 32-channel conditioning.

    Pass prompt/prompt_embeds, seeded generator, steps, etc. as pipeline kwargs.
    Set pixel_mask to zeros for the first all-known single-image pilot. Geometry
    is determined by pixel_mask and must agree with conditioning_latents. No
    RGB source video or reference is accepted: all appearance-bearing control
    comes from conditioning_latents. Cached prompt embeddings can avoid loading
    the large T5 encoder during renderer inference. This wrapper routes cached
    embeddings through encode_prompt to work around the pinned upstream public
    API's string-prompt/cached-embedding conflict. Pass the actual prompt string
    for provenance, or omit it (a never-tokenized empty string is used).

    Both halves must be learned/cached from the same mask convention. A mask
    change without changing the two VAE streams is not a valid control payload.
    """
    _check_pipeline(pipe)
    _validate_latent(conditioning_latents, channels=32)
    _, _, latent_frames, latent_height, latent_width = conditioning_latents.shape
    frames, height, width = 4 * (latent_frames - 1) + 1, 8 * latent_height, 8 * latent_width
    if height % 16 or width % 16:
        raise ValueError("Decoded H,W must be divisible by 16 for transformer patching.")
    _validate_mask(pixel_mask, frames, height, width)
    for key in ("video", "mask", "reference_images"):
        if pipeline_kwargs.get(key) is not None:
            raise ValueError(f"Do not pass {key}; this call uses direct latent conditions.")
        pipeline_kwargs.pop(key, None)
    for key, value in {"num_frames": frames, "height": height, "width": width, "num_videos_per_prompt": 1}.items():
        if key in pipeline_kwargs and pipeline_kwargs[key] != value:
            raise ValueError(f"{key} must equal {value} for this control tensor.")
        pipeline_kwargs[key] = value
    with _temporary_direct_condition(pipe, conditioning_latents, pixel_mask):
        if pipeline_kwargs.get("prompt_embeds") is not None:
            return _cached_prompt_call(pipe, video=None, mask=None, reference_images=None, **pipeline_kwargs)
        return pipe(video=None, mask=None, reference_images=None, **pipeline_kwargs)


def run_smoke_tests():
    """CPU tests requiring torch only; no weights, network or image generation."""
    vae = SimpleNamespace(config=SimpleNamespace(z_dim=16, latents_mean=[0.25] * 16, latents_std=[2.0] * 16))
    raw = torch.randn(1, 16, 1, 2, 2)
    normalized = normalize_wan_latents(vae, raw)
    torch.testing.assert_close(denormalize_wan_latents(vae, normalized), raw)
    empty = normalize_wan_latents(vae, torch.full_like(raw, 0.75))
    condition = assemble_all_known_condition(normalized, empty)
    assert torch.count_nonzero(condition[:, 16:]).item() == empty.numel()

    class FakePipeline:
        vae = SimpleNamespace(dtype=torch.float32)

        def prepare_video_latents(self, *args, **kwargs):
            raise AssertionError("RGB encoder must not run inside direct injection")

        def prepare_masks(self, mask, reference_images=None, generator=None):
            self.seen_mask = mask.clone()
            # Sentinel official-path output: the override must pass this through.
            return torch.full((1, 64, 1, 2, 2), 0.375)

        def encode_prompt(self, *args, **kwargs):
            raise AssertionError("T5 encoder must not run for cached prompts")

        def __call__(self, prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
                     max_sequence_length=512, **kwargs):
            # The two conflicting upstream v0.35.1 public validation checks.
            if not isinstance(prompt, str):
                raise ValueError("Passing a list of prompts is not yet supported.")
            if prompt is not None and prompt_embeds is not None:
                raise ValueError("Cannot forward both prompt and prompt_embeds.")
            assert prompt_embeds is None and negative_prompt_embeds is None
            return self.encode_prompt(prompt, max_sequence_length=max_sequence_length, device=torch.device("cpu"))

    pipe = FakePipeline()
    mask = torch.zeros(1, 1, 1, 16, 16)
    video_placeholder = torch.zeros(1, 3, 1, 16, 16)
    original = pipe.prepare_video_latents.__func__
    try:
        with _temporary_direct_condition(pipe, condition, mask):
            got = pipe.prepare_video_latents(video_placeholder, torch.ones_like(video_placeholder), [[]])
            torch.testing.assert_close(got, condition)
            packed = pipe.prepare_masks(torch.ones_like(video_placeholder), [[]])
            torch.testing.assert_close(pipe.seen_mask, mask)
            assert (packed == 0.375).all().item()
            assert torch.cat([got, packed], dim=1).shape == (1, 96, 1, 2, 2)
            raise RuntimeError("deliberate smoke-test failure")
    except RuntimeError as exc:
        assert str(exc) == "deliberate smoke-test failure"
    assert pipe.prepare_video_latents.__func__ is original
    assert "prepare_video_latents" not in pipe.__dict__
    assert "prepare_masks" not in pipe.__dict__
    assert not hasattr(pipe, "_day11_direct_condition_active")
    positive, negative = torch.randn(1, 128, 4096), torch.randn(1, 128, 4096)
    original_encode_prompt = pipe.encode_prompt.__func__
    got_positive, got_negative = _cached_prompt_call(
        pipe, prompt="Cached scene prompt", prompt_embeds=positive, negative_prompt_embeds=negative
    )
    torch.testing.assert_close(got_positive, positive)
    torch.testing.assert_close(got_negative, negative)
    assert pipe.encode_prompt.__func__ is original_encode_prompt
    assert "encode_prompt" not in pipe.__dict__
    assert not hasattr(pipe, "_day11_cached_prompt_active")
    try:
        with _temporary_cached_prompt(pipe, positive, negative):
            raise RuntimeError("deliberate prompt smoke-test failure")
    except RuntimeError as exc:
        assert str(exc) == "deliberate prompt smoke-test failure"
    assert pipe.encode_prompt.__func__ is original_encode_prompt
    for bad in (torch.zeros(1, 31, 1, 2, 2), torch.full((1, 32, 1, 2, 2), float("nan"))):
        try:
            _validate_latent(bad, channels=32)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed conditioning was accepted")
    return {"normalization_roundtrip": True, "nonzero_empty_stream": True,
            "direct_injection": True, "official_mask_delegation": True,
            "exception_cleanup": True, "malformed_latent_rejection": True,
            "cached_prompt_validation_workaround": True, "cached_prompt_cleanup": True,
            "gpu_inference_tested": False}


def run_upstream_call_smoke_tests():
    """Run the actual pinned __call__ with small model/scheduler stubs on CPU.

    Requires diffusers==0.35.1 but downloads no weights. Reproduces both public
    cached-prompt failures, then checks both RGB and direct-control paths across
    two real upstream loop iterations with CFG. This is interface verification,
    not a claim about actual pretrained renderer quality or GPU compatibility.
    """
    if version("diffusers") != DIFFUSERS_VERSION:
        raise RuntimeError(f"This test requires diffusers=={DIFFUSERS_VERSION}.")
    from diffusers import WanVACEPipeline

    class Transformer:
        dtype = torch.float32
        config = SimpleNamespace(patch_size=(1, 2, 2), vace_layers=[0], in_channels=16)

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return (torch.zeros_like(kwargs["hidden_states"]),)

    class Scheduler:
        order = 1

        def set_timesteps(self, steps, device):
            self.timesteps = torch.arange(steps, device=device)

        def step(self, noise, timestep, latents, return_dict=False):
            return (latents,)

    class Pipeline:
        __call__ = WanVACEPipeline.__call__
        check_inputs = WanVACEPipeline.check_inputs
        prepare_masks = WanVACEPipeline.prepare_masks
        guidance_scale = WanVACEPipeline.guidance_scale
        do_classifier_free_guidance = WanVACEPipeline.do_classifier_free_guidance
        interrupt = WanVACEPipeline.interrupt
        attention_kwargs = WanVACEPipeline.attention_kwargs
        _callback_tensor_inputs = ["latents"]
        _execution_device = torch.device("cpu")
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8
        vae = SimpleNamespace(dtype=torch.float32)

        def __init__(self):
            self.transformer, self.scheduler = Transformer(), Scheduler()

        def encode_prompt(self, *args, **kwargs):
            raise AssertionError("T5 must not run for cached prompts")

        def preprocess_conditions(self, *args):
            return torch.zeros(1, 3, 1, 16, 16), torch.ones(1, 1, 1, 16, 16), [[]]

        def prepare_video_latents(self, *args):
            return torch.full((1, 32, 1, 2, 2), 0.25)

        def prepare_latents(self, *args):
            return torch.zeros(1, 16, 1, 2, 2)

        def maybe_free_model_hooks(self):
            pass

        @contextmanager
        def progress_bar(self, total):
            yield SimpleNamespace(update=lambda: None)

    pipe = Pipeline()
    positive, negative = torch.rand(1, 128, 4096), torch.rand(1, 128, 4096)
    settings = dict(prompt="Cached scene", prompt_embeds=positive, negative_prompt_embeds=negative,
                    max_sequence_length=128, height=16, width=16, num_frames=1,
                    num_inference_steps=2, guidance_scale=5.0, return_dict=False, output_type="latent")
    for bad in (dict(prompt=None, prompt_embeds=positive), dict(prompt="Cached scene", prompt_embeds=positive)):
        try:
            pipe(height=16, width=16, num_frames=1, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected the pinned upstream public-API conflict")
    _cached_prompt_call(pipe, **settings)
    assert len(pipe.transformer.calls) == 4
    for index, call in enumerate(pipe.transformer.calls):
        torch.testing.assert_close(call["encoder_hidden_states"], positive if index % 2 == 0 else negative)
        assert call["control_hidden_states"].shape == (1, 96, 1, 2, 2)
    pipe.transformer.calls.clear()
    condition, mask = torch.randn(1, 32, 1, 2, 2), torch.zeros(1, 1, 1, 16, 16)
    with _temporary_direct_condition(pipe, condition, mask):
        _cached_prompt_call(pipe, **settings)
    assert len(pipe.transformer.calls) == 4
    for call in pipe.transformer.calls:
        torch.testing.assert_close(call["control_hidden_states"][:, :32], condition)
        assert not call["control_hidden_states"][:, 32:].any().item()
    assert all(name not in pipe.__dict__ for name in ("encode_prompt", "prepare_video_latents", "prepare_masks"))
    return {"actual_upstream_call": True, "original_conflicts_reproduced": True,
            "cached_RGB_and_direct_paths": True, "CFG_and_control_payload_checked": True,
            "denoising_steps_per_path": 2, "real_weights_or_GPU_tested": False}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run_smoke_tests(), indent=2))
    if "--upstream" in sys.argv:
        print(json.dumps(run_upstream_call_smoke_tests(), indent=2))
