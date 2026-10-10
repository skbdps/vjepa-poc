"""Scientific displays for Day 8's frozen-probe intervention experiment.

Every probe image is a 24x24 patch-mean RGB readout enlarged by nearest neighbor,
not a generated video. The comparison scenes and model seed are fixed before
seeing results: scene11200 (familiar shift), scene11208 (held-out magnitude),
and learned_residual_1801. No best-scene or best-seed selection is allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import data
except ImportError:
    import data

FIXED_SCENE_SEEDS = (11200, 11208)
FIXED_MODEL_SEED = 1801
FIXED_LEARNED_METHOD = "learned_residual_1801"
FRAME_FOR_STILL = 15
BACKGROUND = (243, 246, 250)
INK = (28, 43, 64)
MUTED = (78, 94, 115)
ACCENT = (14, 110, 144)


def _font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _rgb_image(rgb: np.ndarray, size: int) -> Image.Image:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB display input must be [H,W,3]")
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        raise ValueError("RGB display must be float[0,1] or uint8")
    # Deliberately expose coarse blocks; smoothing could imply nonexistent
    # decoder resolution and hide the actual information in the readout.
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def _place_panel(canvas: Image.Image, xy: tuple[int, int], rgb: np.ndarray,
                 title: str, subtitle: str, size: int) -> None:
    x, y = xy
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), title, font=_font(19, True), fill=INK)
    draw.text((x, y + 26), subtitle, font=_font(15), fill=MUTED)
    canvas.paste(_rgb_image(rgb, size), (x, y + 54))
    draw.rectangle((x, y + 54, x + size - 1, y + 54 + size - 1), outline=(189, 202, 215), width=1)


def _comparison_canvas(spec: dict, pair: dict, predictions: dict,
                       frame: int, include_residual: bool) -> Image.Image:
    """Draw genuine RGB beside coarse readouts with persistent honest labels."""
    if not 0 <= frame < data.N_FRAMES:
        raise ValueError("Invalid video frame")
    tubelet = frame // data.TUBELET
    panels = [
        (pair["frames_source"][frame], "SOURCE RGB", "Original procedural video"),
        (pair["frames_target"][frame], "REQUESTED TARGET RGB", f"Same ball moved {spec['dx']:+d} px"),
        (predictions["noop"]["rgb"][tubelet], "PROBE OUTPUT: SOURCE", "Frozen readout of genuine source"),
    ]
    if include_residual:
        panels.append((predictions["genuine_target"]["rgb"][tubelet], "PROBE OUTPUT: TARGET",
                       "Frozen readout of genuine target"))
    panels.extend([
        (predictions["naive"]["rgb"][tubelet], "PROBE OUTPUT: NAIVE", "Copied feature tokens + source fill"),
    ])
    if include_residual:
        panels.append((predictions["residual"]["rgb"][tubelet], "PROBE OUTPUT: RESIDUAL",
                       "Background-subtracted transport"))
    panels.append((predictions[FIXED_LEARNED_METHOD]["rgb"][tubelet], "PROBE OUTPUT: LEARNED",
                   f"Residual + learned correction; seed {FIXED_MODEL_SEED}"))
    if not include_residual:
        panels.append((predictions["genuine_target"]["rgb"][tubelet], "PROBE OUTPUT: TARGET",
                       "Frozen readout of genuine target"))
    columns = 4 if include_residual else 3
    size, gap, margin = 300, 22, 28
    rows, header, footer, panel_height = 2, 105, 72, size + 62
    width = margin * 2 + columns * size + (columns - 1) * gap
    height = header + rows * panel_height + gap + footer
    # Both dimensions remain even for H.264 yuv420p.
    canvas = Image.new("RGB", (width + width % 2, height + height % 2), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    regime = "held-out shift magnitude" if spec["shift_regime"] == "heldout_magnitude" else "familiar shift magnitude"
    draw.text((margin, 18), "Does a latent edit express the requested movement?", font=_font(25, True), fill=INK)
    draw.text((margin, 55), f"Fixed scene {spec['seed']} | {regime} | frame {frame + 1}/{data.N_FRAMES} | tubelet {tubelet + 1}/{data.N_STEPS}",
              font=_font(17), fill=MUTED)
    for index, (rgb, title, subtitle) in enumerate(panels):
        xy = (margin + (index % columns) * (size + gap), header + (index // columns) * (panel_height + gap))
        _place_panel(canvas, xy, rgb, title, subtitle, size)
    if include_residual:
        x = margin + 3 * (size + gap)
        y = header + panel_height + gap
        draw.text((x, y), "HOW TO READ THIS", font=_font(19, True), fill=ACCENT)
        notes = ["Top left: genuine source pixels.", "Next: the known desired edit.", "", "Probe panels are 24 x 24 grids,", "enlarged without smoothing.", "Each grid summarizes 2 frames.", "", "A probe is a diagnostic readout.", "These are NOT generated videos.", "", "Scene and seed were fixed", "before opening test results."]
        for line_index, line in enumerate(notes):
            draw.text((x, y + 35 + line_index * 23), line, font=_font(16), fill=MUTED)
    draw.text((margin, height - footer + 14), "PROBE OUTPUT = frozen token-to-patch-mean RGB readout. Coarse colours do not establish full video reconstruction.",
              font=_font(16), fill=INK)
    draw.text((margin, height - footer + 40), "Original RGB is shown per frame; each probe grid is held for its two-frame tubelet. Same readout for every method.",
              font=_font(15), fill=MUTED)
    return canvas


def write_video(frames: Iterable[Image.Image], path: Path, fps: int = 8) -> dict:
    """Encode exact displayed RGB canvases through an installed ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the comparison MP4")
    iterator = iter(frames)
    first = next(iterator)
    width, height = first.size
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        for frame in _prepend(first, iterator):
            if frame.size != first.size or frame.mode != "RGB":
                raise ValueError("Video frames must have constant dimensions and RGB mode")
            process.stdin.write(frame.tobytes())
            count += 1
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        path.unlink(missing_ok=True)
        raise
    finally:
        process.stderr.close()
    return {"frames": count, "fps": fps, "resolution": [width, height], "duration_seconds": count / fps}


def _prepend(first, remaining):
    yield first
    yield from remaining


METHOD_ORDER = ("noop", "naive", "residual", "wrong_direction", "wrong_object",
                "geometry_only", "learned_residual", "genuine_target")
METHOD_LABELS = {"noop": "No edit", "naive": "Naive copy", "residual": "Residual transport",
                 "wrong_direction": "Wrong direction", "wrong_object": "Wrong object",
                 "geometry_only": "Geometry-only correction", "learned_residual": "Learned correction",
                 "genuine_target": "Genuine target reference"}
COLORS = {"noop": "#738093", "naive": "#5393b0", "residual": "#287991",
          "wrong_direction": "#c7a8a0", "wrong_object": "#b39baa",
          "geometry_only": "#c59a47", "learned_residual": "#204b72", "genuine_target": "#83a88a"}
PLOT_METRICS = (("source_hole_ratio", "Vacated source: latent error", "Ratio to no-edit error"),
                ("destination_ratio", "Destination: latent error", "Ratio to no-edit error"),
                ("selected_centroid_error_px", "Selected ball: position error", "Pixels"))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def summary_plot(test: Path, out: Path) -> dict:
    """Plot authoritative summary means; whiskers are seed ranges, never CIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((test / "summary.json").read_text())
    rows = _read_csv(test / "per_clip.csv")
    records = {}
    fig, axes = plt.subplots(1, 3, figsize=(16.7, 6.4), sharey=True, layout="constrained")
    for axis, (metric, title, xlabel) in zip(axes, PLOT_METRICS):
        values, lower, upper = [], [], []
        records[metric] = {}
        for arm in METHOD_ORDER:
            authoritative = summary["methods"]["all"][arm][metric]
            mean = _numeric(authoritative["mean"])
            if mean is None:
                raise ValueError(f"Cannot plot missing summary mean: {arm}/{metric}")
            selected = [row for row in rows if row["arm"] == arm]
            seeds = sorted({int(row["seed"]) for row in selected})
            seed_means = []
            for seed in seeds:
                observations = [_numeric(row[metric]) for row in selected if int(row["seed"]) == seed]
                observations = [value for value in observations if value is not None]
                if observations:
                    seed_means.append(float(np.mean(observations)))
            if not seed_means:
                raise ValueError(f"No per-clip observations: {arm}/{metric}")
            # Equal scene sets and seed counts make this mean identical. This
            # check catches a stale CSV paired with a new summary.
            if not np.isclose(mean, np.mean(seed_means), rtol=1e-6, atol=1e-8):
                raise ValueError(f"Summary/CSV disagreement: {arm}/{metric}")
            minimum, maximum = min(seed_means), max(seed_means)
            values.append(mean)
            lower.append(max(0., mean - minimum))
            upper.append(max(0., maximum - mean))
            records[metric][arm] = {"summary_mean": mean, "summary_scene_ci95": authoritative["ci95"],
                                   "n_scenes": authoritative["n_scenes"], "seeds": seeds,
                                   "seed_means": seed_means, "seed_min": minimum, "seed_max": maximum}
        y = np.arange(len(METHOD_ORDER))
        axis.barh(y, values, color=[COLORS[arm] for arm in METHOD_ORDER], height=.62,
                  xerr=np.array([lower, upper]), error_kw={"ecolor": "#12253a", "capsize": 4, "linewidth": 1.5})
        axis.set_yticks(y, [METHOD_LABELS[arm] for arm in METHOD_ORDER], fontsize=10)
        axis.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=14)
        axis.set_xlabel(xlabel + " (lower is better)", fontsize=10)
        axis.grid(axis="x", color="#dde3e9", linewidth=.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        maximum = max(v + e for v, e in zip(values, upper))
        axis.set_xlim(0, max(maximum * 1.25, .2))
        for yi, value, extra in zip(y, values, upper):
            axis.text(value + extra + max(maximum * .015, .002), yi, f"{value:.3g}", va="center", fontsize=9, color="#273d55")
        if metric.endswith("ratio"):
            axis.axvline(1, color="#758195", linewidth=1, linestyle="--", zorder=0)
        else:
            axis.axvline(data.PATCH, color="#758195", linewidth=1, linestyle="--", zorder=0)
    axes[0].invert_yaxis()
    gate = summary["genuine_target_readout_gate"]
    gate_text = "passed" if gate["pass"] else "FAILED: edit semantics inconclusive"
    fig.suptitle("Direct JEPA latent intervention: held-out scenes", x=.17, ha="left", fontsize=17, fontweight="bold")
    fig.supxlabel(f"{summary['n_independent_scenes']} independent scenes. Learned bars average 3 seeds; whiskers show the min–max seed mean, not a confidence interval.\n"
                  f"Dashed lines: no-edit latent error (1) and one-patch position error (16 px). Genuine-target readout gate: {gate_text}.",
                  fontsize=10, color="#465971")
    path = out / "test_summary.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return {"file": path.name, "metric_values": records,
            "means_source": "summary.json methods.all", "whiskers_source": "per_clip.csv mean by seed, then min/max",
            "uncertainty_note": "Whiskers describe seed sensitivity; scene-bootstrap confidence intervals remain in summary.json"}


def load_scene_predictions(test: Path, spec: dict) -> tuple[dict, dict]:
    path = test / "predictions" / (spec["name"] + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        names = archive["method_names"].tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate method names in prediction archive")
        if int(archive["dx_pixels"]) != int(spec["dx"]):
            raise ValueError("Prediction archive displacement mismatch")
        predictions = {name: {key: archive[key][index].astype(np.float32)
                              for key in ("rgb", "occupancy")}
                       for index, name in enumerate(names)}
        auxiliary = {key: archive[key].copy() for key in ("source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target")}
    for name in ("noop", "naive", "residual", FIXED_LEARNED_METHOD, "genuine_target"):
        if name not in predictions:
            raise ValueError(f"Missing predetermined visualization method: {name}")
        if predictions[name]["rgb"].shape != (data.N_STEPS, data.GRID, data.GRID, 3):
            raise ValueError("Unexpected probe RGB dimensions")
        if not np.isfinite(predictions[name]["rgb"]).all():
            raise ValueError("Nonfinite probe RGB values")
    return predictions, auxiliary


def occupancy_plot(spec: dict, predictions: dict, auxiliary: dict, out: Path) -> str:
    """Expose coarse occupancy separately from the RGB probe's limitations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tubelet = FRAME_FOR_STILL // data.TUBELET
    entries = [("Actual target coverage", np.maximum(auxiliary["target_frac"][tubelet], auxiliary["distractor_frac"][tubelet])),
               ("Probe: genuine target", predictions["genuine_target"]["occupancy"][tubelet]),
               ("Probe: source / no edit", predictions["noop"]["occupancy"][tubelet]),
               ("Probe: naive copy", predictions["naive"]["occupancy"][tubelet]),
               ("Probe: residual transport", predictions["residual"]["occupancy"][tubelet]),
               (f"Probe: learned, seed {FIXED_MODEL_SEED}", predictions[FIXED_LEARNED_METHOD]["occupancy"][tubelet])]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8), layout="constrained")
    for axis, (title, values) in zip(axes.flat, entries):
        image = axis.imshow(values, cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                            extent=(0, data.SIZE, data.SIZE, 0))
        axis.set_title(title, fontsize=11)
        axis.set_xticks((0, 192, 384))
        axis.set_yticks((0, 192, 384))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label="Any-ball patch coverage (probe is a soft readout)")
    fig.suptitle(f"Fixed scene {spec['seed']}: occupancy at tubelet {tubelet + 1}/{data.N_STEPS}", fontsize=15, fontweight="bold")
    fig.supxlabel("Coarse diagnostic output; no mask or edit request enters the frozen probe. Same colour scale in every panel.", fontsize=10)
    path = out / (spec["name"] + "_occupancy.png")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(test: str | Path, out: str | Path) -> dict:
    test, out = Path(test), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary_record = summary_plot(test, out)
    specs = {int(spec["seed"]): spec for spec in data.scene_specs("test")}
    scenes = []
    for seed in FIXED_SCENE_SEEDS:
        spec = specs[seed]
        predictions, auxiliary = load_scene_predictions(test, spec)
        pair = data.generate_pair(spec)
        for mask_name, fraction_name in (("masks_source", "source_frac"), ("masks_target", "target_frac"), ("masks_distractor", "distractor_frac")):
            if not np.array_equal(data.patch_fractions(pair[mask_name]), auxiliary[fraction_name]):
                raise ValueError("Regenerated visual scene differs from evaluated masks")
        for frames_name, rgb_name in (("frames_source", "rgb_source"), ("frames_target", "rgb_target")):
            regenerated_rgb = (pair[frames_name].reshape(data.N_STEPS, data.TUBELET, data.GRID, data.PATCH,
                                                       data.GRID, data.PATCH, 3).astype(np.float32)
                               .mean(axis=(1, 3, 5)) / 255).astype(np.float32)
            if not np.array_equal(regenerated_rgb, auxiliary[rgb_name]):
                raise ValueError("Regenerated RGB differs from evaluated patch-mean targets")
        still_path = out / (spec["name"] + "_comparison.png")
        _comparison_canvas(spec, pair, predictions, FRAME_FOR_STILL, True).save(still_path)
        video_path = out / (spec["name"] + "_comparison.mp4")
        video_record = write_video((_comparison_canvas(spec, pair, predictions, frame, False)
                                    for frame in range(data.N_FRAMES)), video_path)
        occupancy_path = occupancy_plot(spec, predictions, auxiliary, out)
        scenes.append({"scene": spec["name"], "scene_seed": seed, "shift_pixels": spec["dx"],
                       "shift_regime": spec["shift_regime"], "learned_method": FIXED_LEARNED_METHOD,
                       "still_frame_zero_based": FRAME_FOR_STILL,
                       "files": [still_path.name, video_path.name, occupancy_path],
                       "video": video_record,
                       "prediction_archive_sha256": sha256(test / "predictions" / (spec["name"] + ".npz"))})
    files = [summary_record["file"], *(name for scene in scenes for name in scene["files"])]
    record = {"version": "day8_fixed_scientific_visuals_v1", "summary": summary_record, "scenes": scenes,
              "scope": "Frozen probe diagnostic, not generated/reconstructed full-resolution video",
              "probe_visualization": "24x24 RGB patch means enlarged with nearest-neighbor; held two source frames",
              "selection_policy": "Fixed first familiar-shift and first held-out-shift scenes; fixed model seed1801; fixed still frame15",
              "source_summary_sha256": sha256(test / "summary.json"),
              "source_per_clip_sha256": sha256(test / "per_clip.csv"),
              "artifacts": [{"path": name, "sha256": sha256(out / name), "bytes": (out / name).stat().st_size} for name in files]}
    (out / "visualization_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="Directory containing summary.json, per_clip.csv, predictions/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.test, args.out)
    print(json.dumps({"status": "complete", "scenes": [row["scene"] for row in result["scenes"]],
                      "artifacts": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
