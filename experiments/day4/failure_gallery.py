"""Post-test diagnostic videos, deliberately selected to preserve failures.

This script does not fit models, tune thresholds, or create a new evaluation.
It selects (1) the lowest SAM2 visible-localization clip and (2) the crossing clip
with most wrong-car outputs from the development-selected V-JEPA variant.
Only actual saved predictions are drawn as predictions. Ground truth supplies a
separately labeled frame-visibility reference and the existing diagnostic scores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import benchmark as bench
import compare_results as comparison

SHORT_NAMES = {"car_A.front_door": "A.door", "car_A.window": "A.window",
               "car_B.front_door": "B.door", "car_B.window": "B.window"}


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["scene"]].append(row)
    return dict(result)


def select_diagnostics(sam_rows, vjepa_rows):
    """Deterministic post-test selection, with explicit rates and tie breaks."""
    sam_groups, vjepa_groups = _groups(sam_rows), _groups(vjepa_rows)
    sam_summaries = {name: bench.summarize_rows(rows) for name, rows in sam_groups.items()}
    vjepa_summaries = {name: bench.summarize_rows(rows) for name, rows in vjepa_groups.items()}
    candidates = [name for name, summary in sam_summaries.items()
                  if summary["localization_accuracy_given_visible"]["denominator"]]
    if not candidates:
        raise ValueError("No SAM2 clip has visible scored targets")
    worst_sam = min(candidates, key=lambda name: (
        sam_summaries[name]["localization_accuracy_given_visible"]["rate"], name))
    crossing = [name for name, rows in vjepa_groups.items() if rows[0]["condition"] == "crossing"]
    if not crossing:
        raise ValueError("No crossing clips in selected V-JEPA outputs")
    worst_crossing = min(crossing, key=lambda name: (
        -vjepa_summaries[name]["wrong_car_given_visible"]["numerator"], name))
    return [
        {"slug": "sam2_lowest_visible_localization", "scene": worst_sam,
         "selection": "Lowest SAM2 visible-localization rate across all test clips; lexicographic scene tie break.",
         "focus_method": sam_rows[0]["method"],
         "selection_summary": sam_summaries[worst_sam],
         "title": "Post-test diagnostic: lowest SAM2 visible localization"},
        {"slug": "vjepa_crossing_wrong_car", "scene": worst_crossing,
         "selection": "Most wrong-car outputs among crossing clips for the development-selected V-JEPA variant; lexicographic scene tie break.",
         "focus_method": vjepa_rows[0]["method"],
         "selection_summary": vjepa_summaries[worst_crossing],
         "title": "Post-test diagnostic: most V-JEPA crossing identity errors"}]


def _overlay_masks(frame, masks):
    canvas = np.asarray(frame, np.float32).copy()
    for ki, color in enumerate(bench.COLORS):
        canvas[masks[ki]] = .60*canvas[masks[ki]]+.40*np.asarray(color)
    image = Image.fromarray(np.rint(canvas).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for ki, color in enumerate(bench.COLORS):
        contours, _ = cv2.findContours(masks[ki].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [tuple(map(int, p)) for p in contour[:, 0]]
            if len(points) >= 2:
                draw.line(points+[points[0]], fill=tuple(color), width=2)
    return image


def _draw_patch_predictions(image, rows, frame_zero=False):
    """Draw only emitted saved predictions; never substitute annotated masks."""
    draw = ImageDraw.Draw(image)
    small = _font(10)
    if frame_zero:
        draw.text((8, 8), "First tubelet; omitted from patch scoring", font=small,
                  fill="white", stroke_width=1, stroke_fill="black")
        return image
    for ki, target in enumerate(bench.PART_NAMES.values()):
        row = rows[target]
        color = tuple(bench.COLORS[ki])
        if row["present"]:
            x, y = row["col"]*bench.PATCH, row["row"]*bench.PATCH
            draw.rectangle((x, y, x+bench.PATCH-1, y+bench.PATCH-1), outline=color, width=3)
            text = SHORT_NAMES[target]
            draw.text((min(x+18, bench.SIZE-66), max(0, y-3)), text,
                      font=small, fill=color, stroke_width=1, stroke_fill="black")
        else:
            draw.text((5, 5+ki*14), SHORT_NAMES[target]+": no patch", font=small,
                      fill=color, stroke_width=1, stroke_fill="black")
    return image


def _index_tubelets(rows):
    return {(row["tubelet"], row["target"]): row for row in rows}


def _diagnostic_frame(rows, kind):
    per_step = defaultdict(list)
    for row in rows:
        per_step[row["tubelet"]].append(row)
    if kind == "vjepa_crossing_wrong_car":
        step = min(per_step, key=lambda t: (-sum(r["wrong_car"] for r in per_step[t]), t))
    else:
        visible = {t: [r for r in rs if r["state"] == "visible"] for t, rs in per_step.items()}
        visible = {t: rs for t, rs in visible.items() if rs}
        step = min(visible, key=lambda t: (sum(r["hit"] for r in visible[t])/len(visible[t]),
                                          -sum(not r["hit"] for r in visible[t]), t))
    return step*bench.TUBELET


def render_frames(scene, sam_masks, sam_rows, vjepa_rows, selected_method, title):
    """Two panels: raw SAM2 masks plus scored patches; selected V-JEPA patches."""
    if sam_masks.shape != (len(scene.frames), 4, bench.SIZE, bench.SIZE) or sam_masks.dtype != bool:
        raise ValueError("Saved SAM2 masks have the wrong shape or dtype")
    sam_by_step, vjepa_by_step = _index_tubelets(sam_rows), _index_tubelets(vjepa_rows)
    width, height = bench.SIZE*2, bench.SIZE+110
    font, small = _font(12), _font(10)
    frames = []
    for index, rgb in enumerate(scene.frames):
        t = index//bench.TUBELET
        sam_at = {target: sam_by_step[(t, target)] for target in bench.PART_NAMES.values()} if t else {}
        vjepa_at = {target: vjepa_by_step[(t, target)] for target in bench.PART_NAMES.values()} if t else {}
        left = _draw_patch_predictions(_overlay_masks(rgb, sam_masks[index]), sam_at, frame_zero=t == 0)
        right = _draw_patch_predictions(Image.fromarray(rgb).copy(), vjepa_at, frame_zero=t == 0)
        canvas = Image.new("RGB", (width, height), (18, 21, 27))
        canvas.paste(left, (0, 64))
        canvas.paste(right, (bench.SIZE, 64))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), title, fill="white", font=font)
        draw.text((8, 22), f"{scene.condition} | frame {index}/{len(scene.frames)-1} | {scene.name}",
                  fill=(190, 199, 212), font=small)
        draw.text((8, 44), "SAM2 raw masks + calibrated scored patch", fill="white", font=font)
        draw.text((bench.SIZE+8, 44), f"V-JEPA {selected_method}: calibrated patch", fill="white", font=font)
        visible, hidden = [], []
        for pid, target in bench.PART_NAMES.items():
            (visible if np.any(scene.masks[index] == pid) else hidden).append(SHORT_NAMES[target])
        reference = f"GT frame visibility: visible {', '.join(visible) or 'none'} | hidden {', '.join(hidden) or 'none'}"
        draw.text((8, bench.SIZE+67), reference, fill=(201, 208, 219), font=small)
        if t:
            def status(part_rows):
                targets = [r for r in part_rows.values() if r["state"] == "visible"]
                return f"visible hits {sum(r['hit'] for r in targets)}/{len(targets)}; wrong car {sum(r['wrong_car'] for r in targets)}"
            draw.text((8, bench.SIZE+81), "Pair score: "+status(sam_at), fill="white", font=small)
            draw.text((bench.SIZE+8, bench.SIZE+81), "Pair score: "+status(vjepa_at), fill="white", font=small)
        for ki, target in enumerate(bench.PART_NAMES.values()):
            draw.text((8+ki*190, bench.SIZE+96), f"Initial ID: {SHORT_NAMES[target]}",
                      fill=tuple(bench.COLORS[ki]), font=small)
        frames.append(np.asarray(canvas))
    return frames


def make_gallery(vjepa_csv, sam2_csv, sam2_clips_dir, out_dir, frozen_config_path=None):
    all_vjepa = comparison.read_prediction_rows(vjepa_csv)
    sam_rows = comparison.read_prediction_rows(sam2_csv)
    if {r["method"] for r in sam_rows} != {"sam2_1_tiny"}:
        raise ValueError("Expected completed SAM2.1-tiny predictions")
    frozen, frozen_path = comparison._load_frozen(vjepa_csv, frozen_config_path)
    selected = frozen["selected_method"]
    vjepa_index = comparison._index_method(all_vjepa, selected)
    sam_index = comparison._index_method(sam_rows, "sam2_1_tiny")
    comparison._check_equal_labels(vjepa_index, sam_index, "sam2_1_tiny")
    if {r["threshold"] for r in vjepa_index.values()} != {frozen["thresholds"][selected]}:
        raise ValueError("Selected V-JEPA threshold does not match the freeze")
    vjepa_rows = list(vjepa_index.values())
    selections = select_diagnostics(sam_rows, vjepa_rows)
    if selections[0]["scene"] == selections[1]["scene"]:
        # One clip can satisfy both criteria; do not save duplicate videos.
        selections = [{**selections[0], "slug": "shared_sam2_vjepa_failure",
            "selection": selections[0]["selection"]+" Also: "+selections[1]["selection"],
            "title": "Post-test diagnostic: SAM2 localization and V-JEPA crossing",
            "additional_selection_summary": selections[1]["selection_summary"]}]
    sam_groups, vjepa_groups = _groups(sam_rows), _groups(vjepa_rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    completed = []
    for selection in selections:
        name = selection["scene"]
        first = sam_groups[name][0]
        scene = bench.generate_scene(first["seed"], first["condition"], name)
        clip_folder = Path(sam2_clips_dir)/name
        mask_path = clip_folder/"sam2_masks.npz"
        manifest = json.loads((clip_folder/"clip_manifest.json").read_text())
        if manifest["original_rgb_sha256"] != hashlib.sha256(scene.frames.tobytes()).hexdigest():
            raise ValueError(f"Regenerated RGB does not match SAM2 input source for {name}")
        if manifest["predicted_masks_sha256"] != comparison._sha256(mask_path):
            raise ValueError(f"Saved SAM2 mask hash mismatch for {name}")
        with np.load(mask_path) as archive:
            masks, ids = archive["masks"], archive["ids"].tolist()
        if ids != list(bench.PART_NAMES):
            raise ValueError("Unexpected predicted-mask initial-ID order")
        frames = render_frames(scene, masks, sam_groups[name], vjepa_groups[name], selected, selection["title"])
        video = out/(selection["slug"]+".mp4")
        bench.write_video(video, frames, fps=12)
        focus = sam_groups[name] if selection["focus_method"] == "sam2_1_tiny" else vjepa_groups[name]
        poster_frame = _diagnostic_frame(focus, selection["slug"])
        poster = out/(selection["slug"]+".png")
        Image.fromarray(frames[poster_frame]).save(poster)
        completed.append({**selection, "video": video.name, "poster": poster.name,
                          "poster_frame": poster_frame, "video_bytes": video.stat().st_size,
                          "sam2_mask_sha256": comparison._sha256(mask_path)})
        del frames, masks
    report = {"purpose": "Post-test diagnostic selection; not a new metric, test, or tuning stage.",
        "development_selected_method": selected, "frozen_config": str(Path(frozen_path).resolve()),
        "predictions": "Colored masks and patch boxes come only from saved model predictions.",
        "raw_mask_caveat": "SAM2 dense masks are ungated; scored patch boxes use the calibrated presence decision.",
        "reference": "GT frame visibility and pair-score text are explicitly labeled references; GT masks are never drawn as predicted masks or edits.",
        "selection_bias": "These clips are intentionally selected failures and are not representative performance estimates.",
        "inputs": {"vjepa_csv_sha256": comparison._sha256(vjepa_csv),
                   "sam2_csv_sha256": comparison._sha256(sam2_csv)},
        "diagnostics": completed, "total_video_bytes": sum(item["video_bytes"] for item in completed)}
    (out/"failure_gallery.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Post-test failure diagnostics", "",
             "These examples were intentionally selected after scoring to preserve failures alongside successful demonstrations. They are not a new test or representative sample.", "",
             f"V-JEPA uses the development-selected `{selected}` configuration. SAM2 raw masks and calibrated patch predictions are shown separately from explicitly labeled ground-truth visibility references.", ""]
    for item in completed:
        lines.extend([f"- **{item['scene']}**: {item['selection']}",
                      f"  [Video]({item['video']}) · [Diagnostic frame {item['poster_frame']}]({item['poster']})", ""])
    (out/"README.md").write_text("\n".join(lines)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa-csv", required=True)
    parser.add_argument("--sam2-csv", required=True)
    parser.add_argument("--sam2-clips-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-config")
    args = parser.parse_args()
    print(json.dumps(make_gallery(args.vjepa_csv, args.sam2_csv, args.sam2_clips_dir,
                                  args.out, args.frozen_config), indent=2))
