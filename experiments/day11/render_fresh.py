"""Prepared, frozen renderer for the eight-scene fresh evaluation.

Code preparation alone does not open any fresh data. At execution, a published
pretest freeze must bind this implementation, its dependencies, prompt cache,
exact rendering configuration, evaluation configuration and checkpoint. The
completed evaluator must refer to those same frozen bytes.

Only the nine predeclared CNN arms are generated, on all eight scenes. Every
scene receives four oracle parity renders followed by nine arm renders: 104
calls in total. There is no method selection, prompt change, seed search,
precision fallback or condition deduplication after fresh data is opened.
Failures remain recorded. The original development renderer stays unchanged.

Run only after an explicit fresh-evaluation authorization and publication:
  python experiments/day11/render_fresh.py --bundle-root EVALUATION \
    --freeze PRETEST.json --publication PUBLICATION.json \
    --checkpoint-freeze TRAINED_CHECKPOINT_FREEZE.json \
    --prompt-cache PROMPTS.pt --prompt-metadata PROMPTS.json --out OUTPUT
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _source_module(name, filename):
    """Load source definitions by exact file, independent of cwd/sys.path."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_prepare = _source_module("day11_fresh_renderer_prepare", "prepare_fresh.py")
_bridge = _source_module("day11_fresh_renderer_bridge", "vace_bridge.py")
# render.py imports its sibling by a bare name. Scope that one alias while
# loading its definitions, then restore the caller's module namespace.
_sentinel = object()
_previous_bridge = sys.modules.get("vace_bridge", _sentinel)
try:
    sys.modules["vace_bridge"] = _bridge
    _base = _source_module("day11_fresh_renderer_base", "render.py")
finally:
    if _previous_bridge is _sentinel:
        sys.modules.pop("vace_bridge", None)
    else:
        sys.modules["vace_bridge"] = _previous_bridge
del _previous_bridge, _sentinel

test_specs = _prepare.test_specs
MODEL_ID, REVISION, SETTINGS, REGIONS = _base.MODEL_ID, _base.REVISION, _base.SETTINGS, _base.REGIONS
PARITY_TOLERANCE, PROMPT, NEGATIVE_PROMPT = _base.PARITY_TOLERANCE, _base.PROMPT, _base.NEGATIVE_PROMPT
RenderRun = _base.RenderRun
array_sha256, sha256, write_json, rgb_metrics = _base.array_sha256, _base.sha256, _base.write_json, _base.rgb_metrics
DIFFUSERS_VERSION = _bridge.DIFFUSERS_VERSION
assemble_all_known_condition = _bridge.assemble_all_known_condition
cached_prompt_vace_call = _bridge.cached_prompt_vace_call
denormalize_wan_latents = _bridge.denormalize_wan_latents
direct_vace_call = _bridge.direct_vace_call
run_upstream_call_smoke_tests = _bridge.run_upstream_call_smoke_tests

EVALUATION_VERSION = "day11_fresh_bridge_evaluation_v1"
SCENES = tuple(range(14000, 14008))
SCENE_NAMES = {item["seed"]: item["name"] for item in test_specs()}
ARMS = (
    "true_source", "true_target", "absolute_copy_repair", "residual_copy_repair",
    "provenance_copy_copy_repair", "provenance_local_copy_repair",
    "provenance_residual_copy_repair", "provenance_residual_wrong_direction",
    "provenance_residual_shuffled",
)
BUNDLE_ARMS = ("true_source", "true_target") + tuple(
    route + "_" + condition
    for route in ("absolute", "residual", "provenance_copy", "provenance_local", "provenance_residual")
    for condition in ("source", "genuine_target", "copy_repair", "wrong_direction", "shuffled")
)
REQUIRED_SOURCES = (
    "experiments/day11/render_fresh.py", "experiments/day11/render.py",
    "experiments/day11/vace_bridge.py", "experiments/day11/evaluate_fresh.py",
    "experiments/day11/prepare_fresh.py", "experiments/day11/transport.py",
    "experiments/day11/assess_fresh.py", "experiments/day11/qualify_development.py",
)


def rendering_config(prompt_cache_sha256, prompt_metadata_sha256, threads=4):
    """Pure pretest configuration builder; reads no files and loads no models."""
    for digest in (prompt_cache_sha256, prompt_metadata_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Prompt bindings must be lowercase SHA256 digests")
    if threads != 4:
        raise ValueError("The fresh rendering configuration fixes four CPU threads")
    return {
        "version": "day11_fresh_render_configuration_v1", "translator": "cnn",
        "arms": list(ARMS), "test_specs": test_specs(), "calls_per_scene": 13, "total_calls": 104,
        "model_id": MODEL_ID, "model_revision": REVISION,
        "diffusers": DIFFUSERS_VERSION, "transformer_dtype": "float16", "vae_dtype": "float32",
        "scheduler_class": "FlowMatchEulerDiscreteScheduler", "attention": "Diffusers native PyTorch SDPA",
        "settings": dict(SETTINGS), "max_sequence_length": 128,
        "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
        "prompt_cache_sha256": prompt_cache_sha256, "prompt_metadata_sha256": prompt_metadata_sha256,
        "parity_max_abs_tolerance": PARITY_TOLERANCE, "threads": threads,
        "deduplicate_conditions": False, "precision_fallback": False,
    }


def _verify_source_bindings(frozen):
    sources = frozen.get("source_hashes", {})
    if not set(REQUIRED_SOURCES) <= set(sources):
        raise ValueError("Pretest freeze does not bind all rendering and evaluation dependencies")
    for name, expected in sources.items():
        path = (REPO / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(REPO.resolve()):
            raise ValueError("Frozen source path escapes the repository")
        if sha256(path) != expected:
            raise ValueError("Source changed after pretest publication: " + name)


def validate_pretest(args):
    """Validate pretest evidence before reading fresh manifests or NPZ files."""
    frozen = json.loads(Path(args.freeze).read_text())
    receipt = json.loads(Path(args.publication).read_text())
    if frozen.get("test_accessed") is not False or frozen.get("test_specs") != test_specs():
        raise ValueError("Expected the published eight-scene pretest specification")
    _verify_source_bindings(frozen)
    assessment = _source_module("day11_fresh_renderer_assessment", "assess_fresh.py")
    if frozen.get("assessment") != assessment.assessment_config():
        raise ValueError("Assessment rules differ from the published pretest freeze")
    freeze_hash, publication_hash = sha256(args.freeze), sha256(args.publication)
    if (receipt.get("freeze_sha256") != freeze_hash or receipt.get("bytes_equal_to_GitHub") is not True
            or receipt.get("test_encoded_before_verification") is not False
            or len(receipt.get("commit", "")) != 40):
        raise ValueError("A byte-verified publication receipt predating fresh extraction is required")
    config = rendering_config(sha256(args.prompt_cache), sha256(args.prompt_metadata), args.threads)
    if frozen.get("rendering") != config:
        raise ValueError("Rendering settings or prompt cache differ from the published pretest freeze")
    if not isinstance(frozen.get("evaluation"), dict) or not frozen["evaluation"]:
        raise ValueError("The pretest freeze must also bind the evaluation configuration")
    checkpoint_reference = frozen["trained_checkpoint_freeze"]
    checkpoint_path = Path(args.checkpoint_freeze)
    if sha256(checkpoint_path) != checkpoint_reference["sha256"]:
        raise ValueError("Trained checkpoint freeze differs from the published pretest binding")
    trained = json.loads(checkpoint_path.read_text())
    if (trained.get("version") != "day11_genuine_jepa_to_wan_training_v1"
            or trained.get("fresh_test_accessed") is not False or trained.get("test_accessed") is not False):
        raise ValueError("Checkpoint freeze must precede fresh data access")
    models = [row for row in trained["models"] if row["arm"] == "cnn"]
    if len(models) != 1 or models[0]["selected_epoch"] != trained["training_config"]["epochs"]:
        raise ValueError("Only the frozen final-epoch CNN may enter fresh rendering")
    return frozen, {
        "test_freeze_sha256": freeze_hash, "test_publication_sha256": publication_hash,
        "trained_checkpoint_freeze_sha256": checkpoint_reference["sha256"],
        "checkpoint_sha256": models[0]["checkpoint_sha256"], "rendering_config": config,
    }


def validate_evaluation_manifest(manifest, frozen, pretest):
    if (manifest.get("version") != EVALUATION_VERSION or manifest.get("complete") is not True
            or manifest.get("fresh_test_accessed") is not True or manifest.get("test_specs") != test_specs()):
        raise ValueError("Expected the complete evaluator output for all eight frozen fresh scenes")
    for key in ("test_freeze_sha256", "test_publication_sha256", "trained_checkpoint_freeze_sha256"):
        if manifest.get(key) != pretest[key]:
            raise ValueError("Fresh evaluator pretest binding differs: " + key)
    if manifest.get("checkpoint_sha256_by_translator", {}).get("cnn") != pretest["checkpoint_sha256"]:
        raise ValueError("Fresh bundles were evaluated with a different CNN checkpoint")
    if manifest.get("evaluation") != frozen["evaluation"]:
        raise ValueError("Fresh evaluator configuration differs from the pretest freeze")
    if manifest.get("source_hashes") != frozen["source_hashes"]:
        raise ValueError("Fresh evaluator source bindings differ from the pretest freeze")
    if set(manifest.get("arms", {})) != set(BUNDLE_ARMS):
        raise ValueError("Fresh evaluator must retain all 27 analysis arms before the fixed rendering subset")
    if manifest.get("translator_arms") != ["cnn", "linear"]:
        raise ValueError("Fresh evaluator must retain both predeclared translator comparisons")
    rows = manifest.get("scenes", [])
    if [int(row["seed"]) for row in rows] != list(SCENES):
        raise ValueError("Fresh scenes are missing, duplicated or reordered")
    for row, spec in zip(rows, test_specs()):
        if row["scene"] != spec["name"] or row["shift_px"] != spec["dx"]:
            raise ValueError("Fresh scene identity or edit differs from the published specification")


def load_bundle(path, seed, evaluation, pretest):
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(bytes(bundle.pop("metadata_json")).decode("utf-8"))
    if (metadata.get("version") != EVALUATION_VERSION or metadata.get("fresh_test_accessed") is not True
            or metadata.get("translator_arm") != "cnn" or int(metadata["seed"]) != seed
            or metadata.get("scene") != SCENE_NAMES[seed]):
        raise ValueError("Unexpected fresh CNN bundle identity")
    for key in ("test_freeze_sha256", "test_publication_sha256", "trained_checkpoint_freeze_sha256", "checkpoint_sha256"):
        if metadata.get(key) != pretest[key]:
            raise ValueError("Bundle pretest/checkpoint binding differs: " + key)
    for key in ("feature_manifest_sha256", "target_manifest_sha256", "vae"):
        if metadata.get(key) != evaluation.get(key):
            raise ValueError("Bundle evaluator provenance differs: " + key)
    if metadata.get("evaluation") != evaluation.get("evaluation"):
        raise ValueError("Bundle evaluation configuration differs from the frozen evaluator")
    actual_arms = {key.removeprefix("latent__") for key in bundle if key.startswith("latent__")}
    if actual_arms != set(BUNDLE_ARMS) or set(metadata["arms"]) != set(BUNDLE_ARMS):
        raise ValueError("Unexpected fresh analysis-arm catalogue")
    for key in ("source_rgb", "target_rgb"):
        if bundle[key].shape != (384, 384, 3) or bundle[key].dtype != np.uint8:
            raise ValueError("Unexpected RGB shape or precision: " + key)
    for arm in BUNDLE_ARMS:
        value = bundle["latent__" + arm]
        if value.shape != (16, 48, 48) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid normalized condition: " + arm)
        if array_sha256(value) != metadata["latent_sha256"][arm]:
            raise ValueError("Latent bytes differ from evaluator metadata: " + arm)
    empty = bundle["normalized_empty_video_latent"]
    if empty.shape != (1, 16, 1, 48, 48) or empty.dtype != np.float32 or not np.isfinite(empty).all():
        raise ValueError("Invalid normalized empty-video latent")
    for key in ["rgb_mask__" + region for region in REGIONS] + [
            "selected_source_mask", "selected_target_mask", "distractor_mask"]:
        if bundle[key].shape != (384, 384) or not np.isin(bundle[key], [0, 1]).all():
            raise ValueError("Invalid scoring mask: " + key)
        bundle[key] = bundle[key].astype(bool)
    return bundle, metadata


class FreshRenderRun(RenderRun):
    """Same generator call; only validated fresh scene paths replace dev paths."""
    def generate(self, scene, name, bundle, condition=None, rgb_input=None, rendered_source=None):
            if scene not in SCENE_NAMES:
                raise ValueError("Unexpected fresh scene seed")
            key = f"{SCENE_NAMES[scene]}/{name}"
            self.stage = key
            old = self.state["renders"].get(key)
            if old is not None:
                for item in old["files"].values():
                    if sha256(self.out / item["path"]) != item["sha256"]:
                        raise ValueError("Completed render file hash changed: " + item["path"])
                latent = torch.from_numpy(np.load(self.out / old["files"]["latent"]["path"], allow_pickle=False))
                rgb = np.asarray(Image.open(self.out / old["files"]["png"]["path"]).convert("RGB"))
                return latent, rgb
            output_dir = self.out / SCENE_NAMES[scene]
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
    args.transformer_dtype = "float16"  # Fixed in the published fresh configuration.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fatal_stage = "preflight"
    runner = None
    try:
        frozen, pretest = validate_pretest(args)
        fatal_stage = "runtime_preflight"
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
        validate_evaluation_manifest(evaluation_manifest, frozen, pretest)
        selected_rows = {int(row["seed"]): row for row in evaluation_manifest["scenes"]}
        bundle_paths = {seed: Path(args.bundle_root) / "cnn" / "scenes" / (SCENE_NAMES[seed] + ".npz")
                        for seed in SCENES}
        for seed in SCENES:
            reference = selected_rows[seed]["bundles"]["cnn"]
            expected_path = "cnn/scenes/" + SCENE_NAMES[seed] + ".npz"
            if reference["path"] != expected_path or sha256(bundle_paths[seed]) != reference["sha256"]:
                raise ValueError("Fresh CNN bundle differs from its evaluator binding")
            bundles[seed], metadata[seed] = load_bundle(bundle_paths[seed], seed, evaluation_manifest, pretest)
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
            "pretest": pretest, "scope": "fresh held-out test under published freeze",
            "arms": list(ARMS), "translator": "cnn",
            "model_id": MODEL_ID, "model_revision": REVISION, "model_files": model_files,
            "settings": SETTINGS, "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "transformer_dtype": args.transformer_dtype, "vae_dtype": "float32", "attention": "Diffusers native PyTorch SDPA",
            "parity_max_abs_tolerance": PARITY_TOLERANCE,
            "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT, "max_sequence_length": 128,
            "prompt_cache_sha256": sha256(args.prompt_cache), "prompt_metadata_sha256": sha256(args.prompt_metadata),
            "evaluation_manifest_sha256": sha256(evaluation_manifest_path),
            "bundles": {str(seed): {"sha256": sha256(bundle_paths[seed]), "metadata": metadata[seed]} for seed in SCENES},
            "source_files": dict(frozen["source_hashes"]),
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
            state = dict(version="day11_fresh_vace_render_v1", fresh_test_accessed=True, created_utc=datetime.now(timezone.utc).isoformat(),
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
        runner = FreshRenderRun(args, pipe, prompt_cache, state)
        runner.checkpoint()
        # Complete all interface gates before rendering any learned arm.
        for seed in SCENES:
            bundle = bundles[seed]
            for view in ("source", "target"):
                runner.stage = f"{SCENE_NAMES[seed]}/prepare_oracle_{view}"
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
                parity_key = f"{SCENE_NAMES[seed]}/{view}"
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
                state["renders"][f"{SCENE_NAMES[seed]}/{arm}"]["arm_metadata"] = metadata[seed]["arms"][arm]
                runner.checkpoint()
        if len(state["renders"]) != 104:
            raise AssertionError("All eight scenes, four parity calls and nine arms must be retained")
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
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--checkpoint-freeze", required=True,
                        help="Relocated copy of the exact trained-checkpoint freeze bound in PRETEST.json")
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--prompt-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--model-cache")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
