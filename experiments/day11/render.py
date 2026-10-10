"""Fixed Day11 development rendering, including official/direct parity gates.

Example:
  python experiments/day11/render.py --bundle-root /content/day11-evaluation \
    --prompt-cache /content/translator_runtime/prompt_embeddings.pt \
    --prompt-metadata /content/translator_runtime/prompt_metadata.json \
    --out /content/day11-render

Runs both predeclared development scenes and every declared arm. A failed
parity check or exception is recorded and stops the run. Precision/scheduler
changes require a new output directory; --resume only resumes identical code,
inputs, model bytes and settings. No renderer weights are trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from PIL import Image
import torch

from vace_bridge import (
    DIFFUSERS_VERSION, MODEL_ID, assemble_all_known_condition,
    cached_prompt_vace_call, denormalize_wan_latents, direct_vace_call,
    run_upstream_call_smoke_tests,
)

REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
SCENES = (13500, 13501)
ARMS = ("true_source", "true_target") + tuple(
    route + "_" + edit for route in ("absolute", "residual")
    for edit in ("source", "genuine_target", "copy_repair", "wrong_direction", "shuffled")
)
REGIONS = ("source_hole", "destination", "distractor", "background", "global")
SETTINGS = dict(height=384, width=384, num_frames=1, num_inference_steps=20,
                guidance_scale=5.0, conditioning_scale=1.0, seed=1111)
PARITY_TOLERANCE = 1e-4
PROMPT = "Two patterned colored balls on a textured background. Static camera, simple geometric scene."
NEGATIVE_PROMPT = "text, watermark, blurry, distorted, extra objects"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_bundle(path, seed):
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(bytes(bundle.pop("metadata_json")).decode("utf-8"))
    if int(metadata["seed"]) != seed:
        raise ValueError("Bundle scene seed is not the predeclared development scene")
    actual_arms = {key.removeprefix("latent__") for key in bundle if key.startswith("latent__")}
    if actual_arms != set(ARMS) or set(metadata["arms"]) != set(ARMS):
        raise ValueError("Bundle must contain exactly all twelve predeclared arms")
    for key in ("source_rgb", "target_rgb"):
        if bundle[key].shape != (384, 384, 3) or bundle[key].dtype != np.uint8:
            raise ValueError(f"Invalid {key} shape or precision")
    for arm in ARMS:
        value = bundle["latent__" + arm]
        if value.shape != (16, 48, 48) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalized condition for {arm}")
    empty = bundle["normalized_empty_video_latent"]
    if empty.shape != (1, 16, 1, 48, 48) or empty.dtype != np.float32 or not np.isfinite(empty).all():
        raise ValueError("Invalid normalized empty-video latent")
    for key in ["rgb_mask__" + region for region in REGIONS] + [
            "selected_source_mask", "selected_target_mask", "distractor_mask"]:
        if bundle[key].shape != (384, 384) or not np.isin(bundle[key], [0, 1]).all():
            raise ValueError(f"Invalid scoring mask {key}")
        bundle[key] = bundle[key].astype(bool)
    return bundle, metadata


def centroid(mask):
    yy, xx = np.nonzero(mask)
    return [float(xx.mean()), float(yy.mean())] if len(xx) else None


def distance(first, second):
    return float(np.linalg.norm(np.asarray(first) - second)) if first is not None and second is not None else None


def rgb_metrics(rgb, bundle, rendered_source=None):
    """Score the saved 8-bit output, not an unrecorded higher-precision image."""
    predicted = rgb.astype(np.float64) / 255
    source, target = (bundle[key].astype(np.float64) / 255 for key in ("source_rgb", "target_rgb"))
    comparisons = {"source_rgb": source, "target_rgb": target}
    if rendered_source is not None:
        comparisons["rendered_true_source"] = rendered_source.astype(np.float64) / 255
    result = {"rgb_mse": {}, "region_pixels": {}}
    for region in REGIONS:
        mask = bundle["rgb_mask__" + region]
        result["region_pixels"][region] = int(mask.sum())
        result["rgb_mse"][region] = {
            name: float(np.square(predicted - reference)[mask].mean()) if mask.any() else None
            for name, reference in comparisons.items()
        }
    # Fixed synthetic-color diagnostic, not a general-purpose object detector.
    # No target-location crop is used to find the generated object.
    selected = bundle["selected_source_mask"]
    distractor = bundle["distractor_mask"]
    background = bundle["rgb_mask__background"]
    if not all(mask.any() for mask in (selected, distractor, background)):
        raise ValueError("Scoring masks require selected, distractor and background pixels")
    selected_color, other_color, background_color = (source[mask].mean(0) for mask in (selected, distractor, background))
    selected_distance = np.linalg.norm(predicted - selected_color, axis=-1)
    other_distance = np.linalg.norm(predicted - other_color, axis=-1)
    background_distance = np.linalg.norm(predicted - background_color, axis=-1)
    detected = ((selected_distance < other_distance) & (selected_distance < background_distance)
                & (selected_distance <= 0.35) & (np.ptp(predicted, axis=-1) >= 0.15))
    predicted_center = centroid(detected)
    target_mask = bundle["selected_target_mask"]
    union = int((detected | target_mask).sum())
    result["selected_color_diagnostic"] = {
        "method": "nearest source object palette versus distractor/background; RGB distance<=0.35 and saturation>=0.15",
        "general_detector": False, "pixels": int(detected.sum()), "centroid_xy": predicted_center,
        "centroid_error_to_source_px": distance(predicted_center, centroid(selected)),
        "centroid_error_to_target_px": distance(predicted_center, centroid(target_mask)),
        "iou_to_target": float((detected & target_mask).sum() / union) if union else None,
        "source_selected_mean_rgb_01": selected_color.tolist(),
        "target_region_mean_rgb_error": float(np.linalg.norm(predicted[target_mask].mean(0) - target[target_mask].mean(0)))
            if target_mask.any() else None,
    }
    return result


class RenderRun:
    def __init__(self, args, pipe, prompt_cache, state):
        self.args, self.pipe, self.prompt_cache, self.state = args, pipe, prompt_cache, state
        self.out = Path(args.out)
        self.stage = "initialization"

    def checkpoint(self):
        write_json(self.out / "metrics.json", self.state)

    def generate(self, scene, name, bundle, condition=None, rgb_input=None, rendered_source=None):
        key = f"dev_{scene}/{name}"
        self.stage = key
        old = self.state["renders"].get(key)
        if old is not None:
            for item in old["files"].values():
                if sha256(self.out / item["path"]) != item["sha256"]:
                    raise ValueError("Completed render file hash changed: " + item["path"])
            latent = torch.from_numpy(np.load(self.out / old["files"]["latent"]["path"], allow_pickle=False))
            rgb = np.asarray(Image.open(self.out / old["files"]["png"]["path"]).convert("RGB"))
            return latent, rgb
        output_dir = self.out / f"dev_{scene}"
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = {key: value for key, value in SETTINGS.items() if key != "seed"}
        settings.update(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
                        prompt_embeds=self.prompt_cache["prompt_embeds"],
                        negative_prompt_embeds=self.prompt_cache["negative_prompt_embeds"],
                        max_sequence_length=128, output_type="latent", return_dict=False,
                        generator=torch.Generator(device="cuda").manual_seed(SETTINGS["seed"]))

        def check_step(pipeline, step, timestep, callback_kwargs):
            if not torch.isfinite(callback_kwargs["latents"]).all().item():
                raise FloatingPointError(f"Nonfinite denoising latent at step {step}, timestep {float(timestep)}")
            return callback_kwargs

        settings["callback_on_step_end"] = check_step
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            if rgb_input is not None:
                latents = cached_prompt_vace_call(
                    self.pipe, video=[Image.fromarray(rgb_input)],
                    mask=[Image.new("RGB", (384, 384), color=0)], **settings
                )[0]
            else:
                latents = direct_vace_call(
                    self.pipe, condition, pixel_mask=torch.zeros(1, 1, 1, 384, 384), **settings
                )[0]
            torch.cuda.synchronize()
            denoise_seconds = time.perf_counter() - started
            if latents.shape != (1, 16, 1, 48, 48) or not torch.isfinite(latents).all().item():
                raise ValueError("Nonfinite or malformed denoised latent")
            latent_cpu = latents.float().cpu()
            raw = denormalize_wan_latents(self.pipe.vae, latents).to(dtype=self.pipe.vae.dtype)
            decoded = self.pipe.vae.decode(raw, return_dict=False)[0]
            if decoded.shape != (1, 3, 1, 384, 384) or not torch.isfinite(decoded).all().item():
                raise ValueError("Nonfinite or malformed decoded image")
            rgb = ((decoded[0, :, 0].float().permute(1, 2, 0).clamp(-1, 1) / 2 + 0.5)
                   .mul(255).round().byte().cpu().numpy())
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        png_path, latent_path = output_dir / (name + ".png"), output_dir / (name + ".npy")
        Image.fromarray(rgb).save(png_path)
        np.save(latent_path, latent_cpu.numpy())
        row = dict(route="official_RGB" if rgb_input is not None else "direct_normalized_VACE_condition",
                   denoise_seconds=denoise_seconds, total_seconds=elapsed,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   input_condition_sha256=array_sha256(condition.cpu().numpy()) if condition is not None else None,
                   input_rgb_sha256=array_sha256(rgb_input) if rgb_input is not None else None,
                   files={"png": {"path": str(png_path.relative_to(self.out)), "sha256": sha256(png_path)},
                          "latent": {"path": str(latent_path.relative_to(self.out)), "sha256": sha256(latent_path)}},
                   metrics=rgb_metrics(rgb, bundle, rendered_source))
        self.state["renders"][key] = row
        self.checkpoint()
        print("RENDER", key, round(elapsed, 3), "seconds", flush=True)
        return latent_cpu, rgb


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fatal_stage = "preflight"
    runner = None
    try:
        if version("diffusers") != DIFFUSERS_VERSION:
            raise RuntimeError(f"Require diffusers=={DIFFUSERS_VERSION}")
        if not torch.cuda.is_available():
            raise RuntimeError("This fixed renderer pilot requires CUDA")
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prompt_cache = torch.load(args.prompt_cache, map_location="cpu", weights_only=True)
        for key in ("prompt_embeds", "negative_prompt_embeds"):
            tensor = prompt_cache[key]
            if tensor.shape != (1, 128, 4096) or not torch.isfinite(tensor).all().item():
                raise ValueError("Prompt cache must contain finite [1,128,4096] tensors")
        prompt_metadata = json.loads(Path(args.prompt_metadata).read_text())
        # Metadata is fully hash-bound; fixed text is additionally included in
        # this run manifest. Any field named prompt/negative_prompt must agree.
        for key, expected in (("prompt", PROMPT), ("negative_prompt", NEGATIVE_PROMPT)):
            if key in prompt_metadata and prompt_metadata[key] != expected:
                raise ValueError("Prompt metadata differs from the fixed pilot text")
        bundles, metadata = {}, {}
        evaluation_manifest_path = Path(args.bundle_root) / "evaluation_manifest.json"
        evaluation_manifest = json.loads(evaluation_manifest_path.read_text())
        if evaluation_manifest.get("complete") is not True:
            raise ValueError("Evaluation bundles must have a completed manifest")
        selected_rows = {int(row["seed"]): row for row in evaluation_manifest["scenes"] if int(row["seed"]) in SCENES}
        if set(selected_rows) != set(SCENES):
            raise ValueError("Evaluation manifest lacks the two predeclared scenes")
        bundle_paths = {seed: Path(args.bundle_root) / "scenes" / f"dev_{seed}.npz" for seed in SCENES}
        for seed in SCENES:
            row = selected_rows[seed]
            if (row["path"] != f"scenes/dev_{seed}.npz" or row.get("renderer_bundle") is not True
                    or sha256(bundle_paths[seed]) != row["sha256"]):
                raise ValueError("Bundle differs from its evaluation-manifest binding")
            bundles[seed], metadata[seed] = load_bundle(bundle_paths[seed], seed)
        interface_test = run_upstream_call_smoke_tests()
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel
        from huggingface_hub import snapshot_download
        fatal_stage = "model_resolution"
        if args.snapshot_dir:
            snapshot = Path(args.snapshot_dir).resolve()
            if snapshot.name != REVISION:
                raise ValueError("Local model snapshot directory must name the exact pinned revision")
        else:
            snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=args.model_cache,
                                             allow_patterns=["transformer/*", "vae/*", "scheduler/*"]))
        model_files = {}
        for component in ("transformer", "vae", "scheduler"):
            files = sorted(path for path in (snapshot / component).rglob("*") if path.is_file())
            if not files:
                raise ValueError("Missing pinned model component: " + component)
            for path in files:
                model_files[str(path.relative_to(snapshot))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        for seed in SCENES:
            provenance = metadata[seed]["vae"]
            if provenance["model_id"] != MODEL_ID or provenance["revision"] != REVISION:
                raise ValueError("Bundle targets were encoded with a different VAE revision")
            for name, expected in provenance["files"].items():
                if model_files["vae/" + name]["sha256"] != expected["sha256"]:
                    raise ValueError("Renderer VAE bytes differ from cached-target VAE bytes")
        bindings = {
            "model_id": MODEL_ID, "model_revision": REVISION, "model_files": model_files,
            "settings": SETTINGS, "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "transformer_dtype": args.transformer_dtype, "vae_dtype": "float32", "attention": "Diffusers native PyTorch SDPA",
            "parity_max_abs_tolerance": PARITY_TOLERANCE,
            "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT, "max_sequence_length": 128,
            "prompt_cache_sha256": sha256(args.prompt_cache), "prompt_metadata_sha256": sha256(args.prompt_metadata),
            "evaluation_manifest_sha256": sha256(evaluation_manifest_path),
            "bundles": {str(seed): {"sha256": sha256(bundle_paths[seed]), "metadata": metadata[seed]} for seed in SCENES},
            "source_files": {name: sha256(Path(__file__).parent / name) for name in ("render.py", "vace_bridge.py")},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "diffusers": version("diffusers"),
                        "transformers": version("transformers"), "gpu": torch.cuda.get_device_name(),
                        "compute_capability": list(torch.cuda.get_device_capability()), "threads": args.threads},
        }
        state_path = out / "metrics.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if not args.resume or state["bindings"] != bindings:
                raise ValueError("Existing output requires --resume with identical inputs/code/settings")
        else:
            state = dict(version="day11_vace_render_v1", created_utc=datetime.now(timezone.utc).isoformat(),
                         bindings=bindings, upstream_interface_smoke_test=interface_test,
                         prompt_metadata=prompt_metadata, parity={}, renders={}, complete=False)
            write_json(state_path, state)
        fatal_stage = "model_loading"
        transformer = WanVACETransformer3DModel.from_pretrained(
            str(snapshot), subfolder="transformer", torch_dtype=getattr(torch, args.transformer_dtype), low_cpu_mem_usage=True
        ).eval().requires_grad_(False).to("cuda")
        vae = AutoencoderKLWan.from_pretrained(
            str(snapshot), subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).eval().requires_grad_(False).to("cuda")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(snapshot), subfolder="scheduler")
        pipe = WanVACEPipeline(transformer=transformer, vae=vae, scheduler=scheduler, tokenizer=None, text_encoder=None)
        pipe.set_progress_bar_config(disable=True)
        state["scheduler_config"] = dict(scheduler.config)
        runner = RenderRun(args, pipe, prompt_cache, state)
        runner.checkpoint()
        # Complete all interface gates before rendering any learned arm.
        for seed in SCENES:
            bundle = bundles[seed]
            for view in ("source", "target"):
                runner.stage = f"dev_{seed}/prepare_oracle_{view}"
                with torch.inference_mode():
                    video, mask, references = pipe.preprocess_conditions(
                        video=[Image.fromarray(bundle[view + "_rgb"])],
                        mask=[Image.new("RGB", (384, 384), color=0)], reference_images=None,
                        batch_size=1, height=384, width=384, num_frames=1,
                        dtype=torch.float32, device=torch.device("cuda"))
                    teacher = pipe.prepare_video_latents(video, mask, references, device=torch.device("cuda")).float().cpu()
                cached = assemble_all_known_condition(
                    torch.from_numpy(bundle["latent__true_" + view])[None, :, None],
                    torch.from_numpy(bundle["normalized_empty_video_latent"]))
                cache_difference = float((teacher - cached).abs().max())
                parity_key = f"dev_{seed}/{view}"
                state["parity"][parity_key] = dict(cached_vs_live_teacher_max_abs=cache_difference,
                                                  tolerance=PARITY_TOLERANCE, passed=False)
                runner.checkpoint()
                if cache_difference > PARITY_TOLERANCE:
                    raise RuntimeError("Cached VAE target differs from official RGB preparation: " + parity_key)
                rgb_latent, _ = runner.generate(seed, "parity_" + view + "_RGB", bundle, rgb_input=bundle[view + "_rgb"])
                direct_latent, _ = runner.generate(seed, "parity_" + view + "_direct", bundle, condition=teacher)
                error = (rgb_latent - direct_latent).abs()
                state["parity"][parity_key].update(denoised_latent_max_abs=float(error.max()),
                    denoised_latent_mean_abs=float(error.mean()), passed=bool(error.max() <= PARITY_TOLERANCE))
                runner.checkpoint()
                if not state["parity"][parity_key]["passed"]:
                    raise RuntimeError("Official RGB/direct denoising parity failed: " + parity_key)
        for seed in SCENES:
            bundle = bundles[seed]
            rendered_source = None
            for arm in ARMS:
                condition = assemble_all_known_condition(
                    torch.from_numpy(bundle["latent__" + arm])[None, :, None],
                    torch.from_numpy(bundle["normalized_empty_video_latent"]))
                _, rgb = runner.generate(seed, arm, bundle, condition=condition, rendered_source=rendered_source)
                if arm == "true_source":
                    rendered_source = rgb
                state["renders"][f"dev_{seed}/{arm}"]["arm_metadata"] = metadata[seed]["arms"][arm]
                runner.checkpoint()
        state["complete"] = True
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        runner.checkpoint()
        print("COMPLETE", state_path, sha256(state_path), flush=True)
    except Exception:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        error_dir = out / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / (stamp + ".txt")
        error_path.write_text(traceback.format_exc())
        write_json(error_dir / (stamp + ".json"), {
            "stage": runner.stage if runner is not None else fatal_stage,
            "error_log": str(error_path.relative_to(out)), "error_log_sha256": sha256(error_path),
            "settings": SETTINGS, "precision_changes_attempted": False,
            "transformer_dtype": args.transformer_dtype,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--prompt-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--model-cache")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--transformer-dtype", choices=("float16", "float32"), default="float16",
                        help="FP32 is a separately named retry with a new --out; FP16 is the first fixed attempt")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
