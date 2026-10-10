"""Fixed scientific displays for the Day 9 temporal source-hole experiment.

Scenes 12200/+32 and 12208/+48, frame 15, all methods, and display policies are
fixed before fresh test access. Probe RGB is a 24x24 diagnostic, not a decoded
video. No smoothing, cherry-picked frame, per-method color scaling, or hidden
heatmap percentile clipping is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DAY8 = HERE.parent / "day8"
sys.path.insert(0, str(DAY8))
import data

_spec = importlib.util.spec_from_file_location("day8_visual_helpers", DAY8 / "visualize.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
write_video = _helpers.write_video
font = _helpers._font
rgb_image = _helpers._rgb_image

FIXED_SCENE_SEEDS = (12200, 12208)
FRAME_FOR_STILL = 15
ARMS = ("noop", "naive", "temporal_mean", "aligned_temporal_mean", "dev_selected", "genuine_target")
REPAIR_ARMS = ("naive", "temporal_mean", "aligned_temporal_mean", "dev_selected")
LABELS = {"noop": "No edit", "naive": "Naive spatial fill", "temporal_mean": "Temporal mean",
          "aligned_temporal_mean": "Aligned temporal mean", "dev_selected": "Dev-selected blend",
          "genuine_target": "Genuine target reference"}
COLORS = {"noop": "#8793a2", "naive": "#56889d", "temporal_mean": "#399e96",
          "aligned_temporal_mean": "#9174b0", "dev_selected": "#204b72", "genuine_target": "#aac298"}
METRICS = (("source_hole_ratio", "Source-hole latent error", "Ratio to no-edit error"),
           ("hole_ghost_mean_occupancy", "Source-hole occupancy", "Mean predicted occupancy"),
           ("hole_rgb_mse", "Source-hole RGB error", "Mean squared error"))
BACKGROUND, INK, MUTED = (243, 246, 250), (28, 43, 64), (78, 94, 115)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_specs():
    return [{"name": f"test_{seed}_dx{dx:+d}", "seed": seed, "dx": dx, "split": "test",
             "shift_regime": regime}
            for seed, dx, regime in ((12200, 32, "seen_magnitude"), (12208, 48, "heldout_magnitude"))]


def selection_label(chosen: dict) -> str:
    kind = "aligned" if chosen["kind"] == "aligned_temporal_mean" else "temporal"
    return f"{kind} blend; alpha = {chosen['alpha']:g}"


def comparison_canvas(spec: dict, pair: dict, predictions: dict, chosen: dict,
                      frame: int, still: bool) -> Image.Image:
    tubelet = frame // data.TUBELET
    panels = [(pair["frames_source"][frame], "SOURCE RGB", "Genuine procedural source pixels"),
              (pair["frames_target"][frame], "TRUE EDITED RGB", f"Ground truth: object shifted {spec['dx']:+d} px")]
    if still:
        panels.extend([(predictions["noop"]["rgb"][tubelet], "PROBE: NO EDIT", "Readout of genuine source features"),
                       (predictions["genuine_target"]["rgb"][tubelet], "PROBE: GENUINE TARGET", "Readout of true edited-video features")])
    panels.append((predictions["naive"]["rgb"][tubelet], "PROBE: NAIVE FILL", "Copied destination + spatial hole fill"))
    if still:
        panels.extend([(predictions["temporal_mean"]["rgb"][tubelet], "PROBE: TEMPORAL MEAN", "Source-video temporal hole donors"),
                       (predictions["aligned_temporal_mean"]["rgb"][tubelet], "PROBE: ALIGNED MEAN", "Donors + estimated context offset")])
    else:
        panels.append((predictions["aligned_temporal_mean"]["rgb"][tubelet], "PROBE: ALIGNED MEAN", "Donors + estimated context offset"))
    panels.append((predictions["dev_selected"]["rgb"][tubelet], "PROBE: DEV-SELECTED", selection_label(chosen)))
    if not still:
        panels.append((predictions["genuine_target"]["rgb"][tubelet], "PROBE: GENUINE TARGET", "Readout of true edited-video features"))
    columns, size, gap, margin = (4 if still else 3), 300, 22, 28
    header, footer, panel_height = 105, 78, size + 62
    width = 2 * margin + columns * size + (columns - 1) * gap
    height = header + 2 * panel_height + gap + footer
    canvas = Image.new("RGB", (width + width % 2, height + height % 2), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    regime = "held-out shift size" if spec["shift_regime"] == "heldout_magnitude" else "familiar shift size"
    draw.text((margin, 18), "Can source-video memory repair the vacated location?", font=font(25, True), fill=INK)
    draw.text((margin, 55), f"Fixed scene {spec['seed']} | {regime} | frame {frame + 1}/32 | tubelet {tubelet + 1}/16",
              font=font(17), fill=MUTED)
    for i, (rgb, title, subtitle) in enumerate(panels):
        x = margin + (i % columns) * (size + gap)
        y = header + (i // columns) * (panel_height + gap)
        draw.text((x, y), title, font=font(18, True), fill=INK)
        draw.text((x, y + 26), subtitle, font=font(14), fill=MUTED)
        canvas.paste(rgb_image(rgb, size), (x, y + 54))
        draw.rectangle((x, y + 54, x + size - 1, y + 54 + size - 1), outline=(189, 202, 215))
    draw.text((margin, height - footer + 11), "PROBE = 24 x 24 patch-mean RGB readout. These are not generated or reconstructed videos.", font=font(15), fill=INK)
    draw.text((margin, height - footer + 37), "No smoothing. A probe grid spans two frames. New methods change only the vacated source hole.", font=font(15), fill=MUTED)
    return canvas


def _plot_modules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _numeric(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def summary_plots(test: Path, out: Path, summary: dict) -> list[dict]:
    """Use evaluator's authoritative scene means and percentile 95% CIs."""
    plt = _plot_modules()
    from matplotlib.ticker import MaxNLocator
    with (test / "per_clip.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    figures = []
    for paired in (False, True):
        arms = ARMS if not paired else ("temporal_mean", "aligned_temporal_mean", "dev_selected")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.7), sharey=True, layout="constrained")
        records = {}
        for axis, (metric, title, xlabel) in zip(axes, METRICS):
            means, lows, highs = [], [], []
            records[metric] = {}
            for arm in arms:
                record = (summary["paired_differences"]["all"][arm + "_minus_naive"][metric] if paired
                          else summary["methods"]["all"][arm][metric])
                mean, ci = _numeric(record["mean"]), record["ci95"]
                if mean is None or any(_numeric(v) is None for v in ci):
                    raise ValueError(f"Missing plot statistic: {arm}/{metric}")
                observations = {row["scene"]: float(row[metric]) for row in rows if row["arm"] == arm}
                if paired:
                    base = {row["scene"]: float(row[metric]) for row in rows if row["arm"] == "naive"}
                    if observations.keys() != base.keys():
                        raise ValueError("Unpaired visualization scene sets")
                    values = [value - base[scene] for scene, value in observations.items()]
                else:
                    values = list(observations.values())
                if len(values) != record["n_scenes"] or not np.isclose(np.mean(values), mean, rtol=1e-7, atol=1e-10):
                    raise ValueError(f"Summary/CSV mismatch: {arm}/{metric}")
                means.append(mean)
                lows.append(mean - ci[0])
                highs.append(ci[1] - mean)
                records[metric][arm] = record
            y = np.arange(len(arms))
            axis.barh(y, means, color=[COLORS[arm] for arm in arms], height=.57,
                      xerr=np.maximum(0, np.array([lows, highs])),
                      error_kw={"ecolor": "#142c44", "capsize": 4, "linewidth": 1.2})
            axis.set_yticks(y, [LABELS[arm] for arm in arms], fontsize=10)
            axis.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
            axis.set_xlabel(("Difference vs naive\n" if paired else "") + xlabel, fontsize=10)
            # Four intervals keep small paired RGB-error tick labels legible.
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.axvline(0, color="#67778b", linewidth=1)
            axis.grid(axis="x", color="#dde3e9", linewidth=.7)
            axis.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
            if not paired:
                axis.set_xlim(left=0)
            axis.margins(x=.22)
            span = max(axis.get_xlim()[1] - axis.get_xlim()[0], 1e-8)
            for yi, value, upper in zip(y, means, highs):
                axis.text(value + max(upper, 0) + .018 * span, yi, f"{value:.3g}", fontsize=9, va="center")
        axes[0].invert_yaxis()
        gate = "passed" if summary["genuine_readout_gate"]["pass"] else "FAILED; edit semantics inconclusive"
        fig.suptitle("Temporal hole repair: " + ("paired changes from naive" if paired else "fresh held-out scenes"),
                     fontsize=17, fontweight="bold", x=.17, ha="left")
        fig.supxlabel(f"{summary['n_independent_scenes']} independent scenes; whiskers = 95% scene-bootstrap intervals (5,000 draws). "
                      + ("Negative differences favor repair." if paired else "Lower values are preferable; genuine target is the readout reference.")
                      + f"\nFrozen selection: {selection_label(summary['chosen'])}. Genuine-target readout gate: {gate}.", fontsize=9.5)
        path = out / ("test_paired_changes.png" if paired else "test_summary.png")
        fig.savefig(path, dpi=100, facecolor="white")
        plt.close(fig)
        figures.append({"file": path.name, "metrics": records, "means_and_ci_source": "summary.json",
                        "independent_unit": "scene", "bootstrap_draws": 5000, "paired": paired})
    return figures


def load_predictions(test: Path, spec: dict):
    path = test / "predictions" / (spec["name"] + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive["method_names"].tolist()) != ARMS or int(archive["dx_pixels"]) != spec["dx"]:
            raise ValueError("Unexpected method order or scene displacement")
        predictions = {arm: {key: archive[key][i].astype(np.float32)
                             for key in ("rgb", "occupancy", "latent_error")}
                       for i, arm in enumerate(ARMS)}
        auxiliary = {key: archive[key].copy() for key in
                     ("source_frac", "target_frac", "distractor_frac", "rgb_source", "rgb_target",
                      "masks_hole", "masks_donor_count", "masks_aligned_donor_count")}
    for arm, prediction in predictions.items():
        if prediction["rgb"].shape != (16, 24, 24, 3) or prediction["occupancy"].shape != (16, 24, 24):
            raise ValueError("Unexpected probe grid shape")
        if not all(np.isfinite(value).all() for value in prediction.values()):
            raise ValueError("Nonfinite visual input")
    return predictions, auxiliary


def verify_rendered_pair(pair: dict, auxiliary: dict):
    for mask, frac in (("masks_source", "source_frac"), ("masks_target", "target_frac"),
                       ("masks_distractor", "distractor_frac")):
        if not np.array_equal(data.patch_fractions(pair[mask]), auxiliary[frac]):
            raise ValueError("Regenerated scene masks differ from evaluated scene")
    for frames, rgb in (("frames_source", "rgb_source"), ("frames_target", "rgb_target")):
        regenerated = (pair[frames].reshape(16, 2, 24, 16, 24, 16, 3).astype(np.float32)
                       .mean(axis=(1, 3, 5)) / 255).astype(np.float32)
        if not np.array_equal(regenerated, auxiliary[rgb]):
            raise ValueError("Regenerated scene pixels differ from evaluated scene")


def occupancy_plot(spec: dict, predictions: dict, auxiliary: dict, out: Path) -> str:
    plt = _plot_modules()
    step = FRAME_FOR_STILL // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.7), layout="constrained")
    for axis, arm in zip(axes.flat, ARMS):
        picture = axis.imshow(predictions[arm]["occupancy"][step], vmin=0, vmax=1, cmap="magma",
                              interpolation="nearest", extent=(0, 384, 384, 0))
        hole = auxiliary["masks_hole"][step]
        # Tiny cyan boxes mark exactly which source-hole tokens may change.
        from matplotlib.patches import Rectangle
        for y, x in np.argwhere(hole):
            axis.add_patch(Rectangle((x * 16, y * 16), 16, 16, fill=False, linewidth=.7, edgecolor="#63d9ef"))
        axis.set_title("Probe: " + LABELS[arm], fontsize=11)
        axis.set_xticks((0, 192, 384))
        axis.set_yticks((0, 192, 384))
    fig.colorbar(picture, ax=axes.ravel().tolist(), shrink=.8, label="Predicted any-ball patch occupancy")
    fig.suptitle(f"Fixed scene {spec['seed']}: coarse occupancy at frame 16 / tubelet 8", fontsize=15, fontweight="bold")
    fig.supxlabel("Cyan boxes: vacated source-hole tokens. Same [0,1] color scale, nearest-neighbor grids.\n"
                  "Genuine-target output is the reference for probe ghosting; every nonzero value is not a real object.", fontsize=10)
    path = out / (spec["name"] + "_occupancy.png")
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    return path.name


def hole_diagnostics(spec: dict, predictions: dict, auxiliary: dict, out: Path) -> dict:
    plt = _plot_modules()
    step = FRAME_FOR_STILL // 2
    hole = auxiliary["masks_hole"][step].astype(bool)
    if not hole.any():
        raise ValueError("Fixed visualization frame has no source-hole cells")
    points = np.argwhere(hole)
    y0, x0 = np.maximum(0, points.min(0) - 2)
    y1, x1 = np.minimum(24, points.max(0) + 3)
    crop = np.s_[y0:y1, x0:x1]
    error_max = max(float(predictions[arm]["latent_error"][step][hole].max()) for arm in REPAIR_ARMS)
    error_max = max(error_max, np.finfo(np.float32).eps)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8), layout="constrained")
    panels = [(LABELS[arm] + "\n1024-D latent MSE", predictions[arm]["latent_error"][step], "inferno", error_max, "MSE")
              for arm in REPAIR_ARMS]
    panels.extend([("Visible temporal donors", auxiliary["masks_donor_count"][step], "viridis", 15, "Donor count"),
                   ("Accepted aligned donors", auxiliary["masks_aligned_donor_count"][step], "viridis", 15, "Donor count"),
                   ("Dev-selected probe occupancy", predictions["dev_selected"]["occupancy"][step], "magma", 1, "Occupancy"),
                   ("Genuine-target probe occupancy", predictions["genuine_target"]["occupancy"][step], "magma", 1, "Occupancy")])
    for axis, (title, values, cmap_name, vmax, label) in zip(axes.flat, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#dce2e8")
        displayed = np.ma.array(values[crop], mask=~hole[crop])
        picture = axis.imshow(displayed, vmin=0, vmax=vmax, cmap=cmap, interpolation="nearest",
                              extent=(int(x0), int(x1), int(y1), int(y0)))
        axis.set_title(title, fontsize=10.5, pad=8)
        axis.set_xlabel("Patch x", fontsize=9)
        axis.set_ylabel("Patch y", fontsize=9)
        axis.tick_params(labelsize=8)
        fig.colorbar(picture, ax=axis, shrink=.72, label=label, pad=.02).ax.tick_params(labelsize=8)
    fig.suptitle(f"Fixed scene {spec['seed']}: source-hole repair at frame 16 / tubelet 8", fontsize=16, fontweight="bold")
    fig.supxlabel("Gray = outside the evaluated source hole. Same crop for every panel; all hole cells included.\n"
                  "Top row shares one un-clipped error scale. Donor count 0 means spatial-fill fallback. Genuine occupancy is the probe reference.", fontsize=9.5)
    path = out / (spec["name"] + "_hole_diagnostics.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    return {"file": path.name, "crop_patch_bounds_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "latent_error_vmin_vmax": [0, error_max], "donor_count_vmin_vmax": [0, 15],
            "occupancy_vmin_vmax": [0, 1], "hole_tokens_at_fixed_tubelet": int(hole.sum()),
            "visible_donor_coverage_at_fixed_tubelet": float(np.mean(auxiliary["masks_donor_count"][step][hole] > 0)),
            "aligned_donor_coverage_at_fixed_tubelet": float(np.mean(auxiliary["masks_aligned_donor_count"][step][hole] > 0))}


def render(test: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((test / "summary.json").read_text())
    if summary["n_independent_scenes"] != 16:
        raise ValueError("Visualizations require the complete 16-scene evaluation")
    plots = summary_plots(test, out, summary)
    scenes = []
    for spec in fixed_specs():
        predictions, auxiliary = load_predictions(test, spec)
        pair = data.generate_pair(spec)
        verify_rendered_pair(pair, auxiliary)
        still = out / (spec["name"] + "_comparison.png")
        comparison_canvas(spec, pair, predictions, summary["chosen"], FRAME_FOR_STILL, True).save(still)
        video = out / (spec["name"] + "_comparison.mp4")
        video_record = write_video((comparison_canvas(spec, pair, predictions, summary["chosen"], frame, False)
                                    for frame in range(32)), video, fps=8)
        occupancy = occupancy_plot(spec, predictions, auxiliary, out)
        holes = hole_diagnostics(spec, predictions, auxiliary, out)
        scenes.append({"scene": spec["name"], "scene_seed": spec["seed"], "shift_pixels": spec["dx"],
                       "shift_regime": spec["shift_regime"], "still_frame_zero_based": FRAME_FOR_STILL,
                       "files": [still.name, video.name, occupancy, holes["file"]],
                       "video": video_record, "hole_diagnostic": holes,
                       "prediction_archive_sha256": sha256(test / "predictions" / (spec["name"] + ".npz"))})
    files = [row["file"] for row in plots] + [name for scene in scenes for name in scene["files"]]
    record = {"version": "day9_fixed_scientific_visuals_v1", "summary_plots": plots, "scenes": scenes,
              "chosen": summary["chosen"], "scope": "Coarse frozen readout diagnostic, not generated/reconstructed full RGB video",
              "display": "24x24 nearest-neighbor RGB grids, held for two source frames; no hidden smoothing or error clipping",
              "selection_policy": "Preselected scenes 12200/+32 and 12208/+48; fixed still frame15; no learned random seeds",
              "source_summary_sha256": sha256(test / "summary.json"), "source_per_clip_sha256": sha256(test / "per_clip.csv"),
              "visual_source_sha256": sha256(Path(__file__)),
              "artifacts": [{"path": name, "sha256": sha256(out / name), "bytes": (out / name).stat().st_size} for name in files]}
    (out / "visualization_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = render(args.test, args.out)
    print(json.dumps({"status": "complete", "artifacts": record["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
