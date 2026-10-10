"""Run the reconstructed 20-call actual-video pilot on the saved Day12 guides.

The official RGB video path performs causal video-VAE encoding. Independent
image latents are never stacked and injected as native video latents.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch
from common import array_sha256, file_record, load_guides, sha256, write_json
from config import (ARMS, DIFFUSERS_VERSION, MAX_SEQUENCE_LENGTH, MODEL_ID,
                    NEGATIVE_PROMPT, NOISE_SEEDS, PROMPTS, REVISION, SCENES, SETTINGS)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "day11"))
from vace_bridge import cached_prompt_vace_call, denormalize_wan_latents


def load_prompts(root):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    expected = dict(version="day12_prompt_cache_reconstructed_v1", complete=True,
                    model_id=MODEL_ID, model_revision=REVISION, encoder_dtype="float32",
                    saved_dtype="float16", max_sequence_length=MAX_SEQUENCE_LENGTH,
                    negative_prompt=NEGATIVE_PROMPT, prompts={str(k): v for k, v in PROMPTS.items()})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("Prompt metadata mismatch: " + key)
    if set(metadata["caches"]) != {str(seed) for seed in SCENES}:
        raise ValueError("Exactly two fixed motion-prompt caches are required")
    prompts = {}
    for seed in SCENES:
        row = metadata["caches"][str(seed)]
        path = root / f"prompt_{seed}.pt"
        if row["file"]["path"] != path.name or sha256(path) != row["file"]["sha256"]:
            raise ValueError("Prompt cache file differs from metadata")
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if set(cache) != {"prompt_embeds", "negative_prompt_embeds"}:
            raise ValueError("Unexpected prompt tensors")
        for key, tensor in cache.items():
            if (tensor.shape != (1, MAX_SEQUENCE_LENGTH, 4096) or tensor.dtype != torch.float16
                    or not torch.isfinite(tensor).all().item()
                    or array_sha256(tensor.numpy()) != row["arrays"][key]):
                raise ValueError("Prompt tensor is malformed or changed: " + key)
        prompts[seed] = cache
    return metadata, prompts


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = "preflight"
    try:
        guides, bundles = load_guides(args.guides)
        prompt_metadata, prompts = load_prompts(args.prompt_dir)
        if version("diffusers") != DIFFUSERS_VERSION or not torch.cuda.is_available():
            raise RuntimeError("Require pinned diffusers and a CUDA runtime")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler,
                               WanVACEPipeline, WanVACETransformer3DModel)
        from huggingface_hub import snapshot_download
        stage = "model_resolution"
        snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=args.model_cache,
                                         allow_patterns=["transformer/*", "vae/*", "scheduler/*"]))
        model_files = {str(p.relative_to(snapshot)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                       for component in ("transformer", "vae", "scheduler")
                       for p in sorted((snapshot / component).rglob("*")) if p.is_file()}
        for name, expected in guides["vae"]["files"].items():
            if model_files["vae/" + name]["sha256"] != expected["sha256"]:
                raise ValueError("Renderer VAE does not match the prepared image guides")
        source_paths = [Path(__file__).parent / name for name in ("render.py", "common.py", "config.py")]
        source_paths.append(Path(__file__).resolve().parents[1] / "day11/vace_bridge.py")
        initial_noise = {seed: torch.randn((1, 16, 3, 48, 48), device="cuda", dtype=torch.float32,
                         generator=torch.Generator(device="cuda").manual_seed(seed)) for seed in NOISE_SEEDS}
        bindings = dict(version="day12_reconstructed_video_render_v1", model_id=MODEL_ID,
                        model_revision=REVISION, model_files=model_files, settings=SETTINGS,
                        noise_seeds=list(NOISE_SEEDS), arms=list(ARMS), scene_seeds=list(SCENES), scenes=list(SCENES),
                        transformer_dtype="float16", vae_dtype="float32", noise_dtype="float32", threads=4,
                        scheduler_class="FlowMatchEulerDiscreteScheduler", max_sequence_length=MAX_SEQUENCE_LENGTH,
                        guide_manifest_sha256=sha256(Path(args.guides) / "manifest.json"),
                        prompt_metadata_sha256=sha256(Path(args.prompt_dir) / "metadata.json"),
                        source_files={str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
                                      for p in source_paths},
                        initial_noise_sha256={str(seed): array_sha256(x.cpu().numpy()) for seed, x in initial_noise.items()},
                        runtime=dict(python=platform.python_version(), torch=torch.__version__,
                                     diffusers=version("diffusers"), transformers=version("transformers"),
                                     gpu=torch.cuda.get_device_name(),
                                     compute_capability=list(torch.cuda.get_device_capability())))
        state_path = out / "metrics.json"
        # Validate existing state before assigning/replacing any of its fields.
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if not args.resume or state.get("bindings") != bindings:
                raise ValueError("Existing results require --resume with identical code, inputs and runtime")
            expected_keys = {f"dev_{s}/noise_{n}/{a}" for s in SCENES for n in NOISE_SEEDS for a in ARMS}
            if not set(state["renders"]).issubset(expected_keys):
                raise ValueError("Existing results contain unexpected render keys")
            for row in state["renders"].values():
                for item in row["files"].values():
                    if sha256(out / item["path"]) != item["sha256"]:
                        raise ValueError("Existing render payload changed")
        else:
            state = dict(version="day12_reconstructed_video_render_v1", bindings=bindings,
                         created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                         route="official_RGB_video_VACE", prompt_metadata=prompt_metadata, renders={})
            write_json(state_path, state)
        stage = "model_loading"
        transformer = WanVACETransformer3DModel.from_pretrained(
            str(snapshot), subfolder="transformer", torch_dtype=torch.float16,
            low_cpu_mem_usage=True).eval().requires_grad_(False).to("cuda")
        vae = AutoencoderKLWan.from_pretrained(str(snapshot), subfolder="vae", torch_dtype=torch.float32,
                                             low_cpu_mem_usage=True).eval().requires_grad_(False).to("cuda")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(snapshot), subfolder="scheduler")
        if "scheduler_config" in state and state["scheduler_config"] != dict(scheduler.config):
            raise ValueError("Existing scheduler configuration changed")
        state["scheduler_config"] = dict(scheduler.config)
        pipe = WanVACEPipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                               tokenizer=None, text_encoder=None)
        pipe.set_progress_bar_config(disable=True)
        write_json(state_path, state)
        control_audit = {}

        def audit_control(module, positional, kwargs):
            control = kwargs["control_hidden_states"]
            if control.shape != (1, 96, 3, 48, 48) or not torch.isfinite(control).all().item():
                raise ValueError("Actual VACE control geometry/values invalid")
            if torch.count_nonzero(control[:, 32:]).item():
                raise ValueError("All-known packed mask must be zero")
            if not control_audit:
                reactive_count = int(torch.count_nonzero(control[:, 16:32]).item())
                if reactive_count == 0:
                    raise ValueError("Encoded reactive stream must not be a zero latent")
                control_audit.update(shape=list(control.shape),
                    sha256=array_sha256(control.detach().cpu().numpy()),
                    reactive_nonzero_elements=reactive_count, packed_mask_zero=True)

        transformer.register_forward_pre_hook(audit_control, with_kwargs=True)

        def check_step(pipeline, step, timestep, callback_kwargs):
            if not torch.isfinite(callback_kwargs["latents"]).all().item():
                raise FloatingPointError(f"Nonfinite denoising latent at step {step}")
            return callback_kwargs

        for scene in SCENES:
            for noise_seed in NOISE_SEEDS:
                for arm in ARMS:
                    key = f"dev_{scene}/noise_{noise_seed}/{arm}"
                    stage = key
                    if key in state["renders"]:
                        continue
                    guide = bundles[scene]["guide_" + arm]
                    control_audit.clear()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        latent = cached_prompt_vace_call(
                            pipe, video=[Image.fromarray(frame) for frame in guide],
                            mask=[Image.new("RGB", (384, 384), color=0) for _ in range(9)],
                            prompt=PROMPTS[scene], negative_prompt=NEGATIVE_PROMPT,
                            **prompts[scene], **SETTINGS, max_sequence_length=MAX_SEQUENCE_LENGTH,
                            latents=initial_noise[noise_seed].clone(), output_type="latent", return_dict=False,
                            generator=torch.Generator(device="cuda").manual_seed(noise_seed),
                            callback_on_step_end=check_step)[0]
                        if latent.shape != (1, 16, 3, 48, 48) or not torch.isfinite(latent).all().item():
                            raise FloatingPointError("Denoised video latent is malformed/nonfinite")
                        torch.cuda.synchronize()
                        denoise_seconds = time.perf_counter() - started
                        raw = denormalize_wan_latents(vae, latent).to(dtype=torch.float32)
                        decoded = vae.decode(raw, return_dict=False)[0]
                        if decoded.shape != (1, 3, 9, 384, 384) or not torch.isfinite(decoded).all().item():
                            raise FloatingPointError("Decoded video is malformed/nonfinite")
                        rgb = ((decoded[0].float().permute(1, 2, 3, 0).clamp(-1, 1) / 2 + .5)
                               .mul(255).round().byte().cpu().numpy())
                        latent_np = latent.float().cpu().numpy()
                    if not control_audit:
                        raise AssertionError("Actual VACE conditioning was not observed")
                    directory = out / key
                    directory.mkdir(parents=True, exist_ok=True)
                    rgb_path, latent_path, gif_path = (directory / n for n in ("rgb.npy", "latent.npy", "preview.gif"))
                    np.save(rgb_path, rgb)
                    np.save(latent_path, latent_np)
                    frames = [Image.fromarray(x) for x in rgb]
                    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=125, loop=0)
                    state["renders"][key] = dict(scene_seed=scene, noise_seed=noise_seed, arm=arm,
                        guide_array_sha256=array_sha256(guide), route="official_RGB_video_VACE",
                        control_audit=dict(control_audit),
                        denoise_seconds=denoise_seconds, total_seconds=time.perf_counter() - started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        files={"rgb": file_record(rgb_path, out), "latent": file_record(latent_path, out),
                               "gif": file_record(gif_path, out)})
                    write_json(state_path, state)
                    print("RENDER", key, flush=True)
        state["complete"] = len(state["renders"]) == 20
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        print("COMPLETE", state_path, flush=True)
    except Exception:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(out / "errors" / (stamp + ".json"), dict(stage=stage, traceback=traceback.format_exc(),
                   precision_changes_attempted=False, code_reconstructed=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guides", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-cache")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
