"""Render the Day11 provenance-transport DEVELOPMENT follow-up.

This separate entry point leaves the initial twelve-arm render.py unchanged.
It renders both previously inspected development scenes (13500,13501), all
twelve original arms and all fifteen predeclared transport arms. This is not a
fresh test or a claim of held-out generalization. The transport operator uses
editor-specific exact JEPA provenance; it is not learned semantic matching.

The generator, noise seed1111, 20steps, CFG5, conditioning1, 384x384 single
frame, prompts, scheduler and parity tolerance are identical to the first run.
Every arm is actually rendered, even when conditions are identical; no visual
selection or condition deduplication is performed. Existing completed outputs
can only resume under identical hash bindings.

Usage matches render.py, with --bundle-root pointing to the completed
transport.py output and a new --out. The core generator runner and RGB metrics
are imported unchanged from render.py. The four RGB/direct oracle parity
comparisons complete before any experimental arm is rendered.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import traceback

import numpy as np
from PIL import Image
import torch

from render import (
    ARMS as ORIGINAL_ARMS, MODEL_ID, REVISION, SCENES, SETTINGS, REGIONS,
    PARITY_TOLERANCE, PROMPT, NEGATIVE_PROMPT, RenderRun, sha256, write_json,
)
from vace_bridge import DIFFUSERS_VERSION, assemble_all_known_condition, run_upstream_call_smoke_tests


TRANSPORT_VERSION = "day11_development_provenance_transport_v1"
TRANSPORT_ROUTES = ("provenance_copy", "provenance_local", "provenance_residual")
CONDITIONS = ("source", "genuine_target", "copy_repair", "wrong_direction", "shuffled")
ARMS = ORIGINAL_ARMS + tuple(route + "_" + condition for route in TRANSPORT_ROUTES for condition in CONDITIONS)
assert len(ARMS) == 27 and len(set(ARMS)) == 27


def validate_followup_manifest(manifest):
    if manifest.get("version") != TRANSPORT_VERSION or manifest.get("fresh_test_accessed") is not False:
        raise ValueError("Expected the explicitly development-only transport follow-up")
    if set(manifest.get("arms", {})) != set(ARMS):
        raise ValueError("Follow-up manifest must declare all 27 fixed arms")
    if manifest.get("renderer_seeds") != list(SCENES):
        raise ValueError("Follow-up rendering must use only development13500/13501")
    if manifest.get("all_development_seeds") != list(range(13500, 13508)):
        raise ValueError("Follow-up evaluation must include all eight existing development scenes")
    rows = manifest.get("scenes", [])
    if [int(row["seed"]) for row in rows] != list(range(13500, 13508)):
        raise ValueError("Follow-up scene evidence is missing, duplicated or reordered")
    expected = manifest.get("source_hashes", {}).get("experiments/day11/transport.py")
    if expected != sha256(Path(__file__).with_name("transport.py")):
        raise ValueError("Transport source differs from the follow-up bundle provenance")


def load_bundle(path, seed):
    """Validate the extended bundle without weakening the original validator."""
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(bytes(bundle.pop("metadata_json")).decode("utf-8"))
    if int(metadata["seed"]) != seed or seed not in SCENES:
        raise ValueError("Bundle is not a predeclared follow-up development scene")
    if metadata.get("version") != TRANSPORT_VERSION or metadata.get("fresh_test_accessed") is not False:
        raise ValueError("Bundle must explicitly identify the development follow-up")
    if metadata.get("transport_source_sha256") != sha256(Path(__file__).with_name("transport.py")):
        raise ValueError("Bundle transport implementation hash differs")
    actual_arms = {key.removeprefix("latent__") for key in bundle if key.startswith("latent__")}
    if actual_arms != set(ARMS) or set(metadata["arms"]) != set(ARMS):
        raise ValueError("Bundle must contain exactly all 27 predeclared arms")
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
        validate_followup_manifest(evaluation_manifest)
        if evaluation_manifest.get("complete") is not True:
            raise ValueError("Evaluation bundles must have a completed manifest")
        selected_rows = {int(row["seed"]): row for row in evaluation_manifest["scenes"] if int(row["seed"]) in SCENES}
        if set(selected_rows) != set(SCENES):
            raise ValueError("Evaluation manifest lacks the two predeclared scenes")
        bundle_paths = {seed: Path(args.bundle_root) / "scenes" / f"dev_{seed}.npz" for seed in SCENES}
        for seed in SCENES:
            row = selected_rows[seed]
            if (row["renderer_path"] != f"scenes/dev_{seed}.npz" or row.get("renderer_bundle") is not True
                    or sha256(bundle_paths[seed]) != row["renderer_sha256"]):
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
            "followup_version": TRANSPORT_VERSION, "fresh_test_accessed": False,
            "scope": "Existing development follow-up; no held-out claim", "arm_catalogue": list(ARMS),
            "deduplicate_conditions": False,
            "model_id": MODEL_ID, "model_revision": REVISION, "model_files": model_files,
            "settings": SETTINGS, "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "transformer_dtype": args.transformer_dtype, "vae_dtype": "float32", "attention": "Diffusers native PyTorch SDPA",
            "parity_max_abs_tolerance": PARITY_TOLERANCE,
            "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT, "max_sequence_length": 128,
            "prompt_cache_sha256": sha256(args.prompt_cache), "prompt_metadata_sha256": sha256(args.prompt_metadata),
            "evaluation_manifest_sha256": sha256(evaluation_manifest_path),
            "bundles": {str(seed): {"sha256": sha256(bundle_paths[seed]), "metadata": metadata[seed]} for seed in SCENES},
            "source_files": {name: sha256(Path(__file__).parent / name) for name in ("render_transport.py", "render.py", "vace_bridge.py", "transport.py")},
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
            state = dict(version="day11_vace_transport_render_v1", scope="existing development follow-up",
                         fresh_test_accessed=False, created_utc=datetime.now(timezone.utc).isoformat(),
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
        if len(state["renders"]) != len(SCENES) * (4 + len(ARMS)):
            raise AssertionError("Every parity path and every declared arm must be retained")
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
                        help="FP32 requires a separately named attempt in a new output directory")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
