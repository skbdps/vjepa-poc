"""Fixed, honest displays of the Day 10 single-image JEPA intervention.

The primary preview has two panels: original image and edited 24x24 RGB probe
readout. A separate evidence view compares readouts with an encoded, genuine
target and the target pixels. Every scene is displayed; no best-case selection
is performed. Motion is a prescribed path, not predicted dynamics, and no
interpolation or smoothing is applied to the feature readout.

Output encoding is completed outside the shared workspace, fully decoded for
validation, then atomically installed. This prevents an incomplete MP4 header
from being exposed as a deliverable while ffmpeg is still running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (244, 247, 250)
INK = (27, 42, 61)
MUTED = (77, 94, 114)
BORDER = (182, 196, 209)
SIZE, MARGIN, GAP = 384, 28, 28
ARMS = ("noop", "copy_repair", "wrong_direction", "genuine_target")
EXPECTED_SEEDS = (13200, 13201, 13202, 13203)
DISPLAY_INDICES = (0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0)


def font(size: int, bold: bool = False):
    choices = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in choices:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_image(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    if array.shape not in ((24, 24, 3), (384, 384, 3)):
        raise ValueError(f"Unexpected RGB shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite RGB display input")
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError("RGB must be float[0,1] or uint8")
    return Image.fromarray(array).resize((SIZE, SIZE), Image.Resampling.NEAREST)


def text(draw, xy, message, *, size=17, bold=False, fill=INK, max_width=None):
    face = font(size, bold)
    if max_width is not None and draw.textlength(message, font=face) > max_width:
        raise ValueError(f"Display text exceeds allotted width: {message}")
    draw.text(xy, message, font=face, fill=fill)


def panel(canvas, xy, rgb, title, subtitle):
    x, y = xy
    draw = ImageDraw.Draw(canvas)
    text(draw, (x, y), title, size=21, bold=True, max_width=SIZE)
    text(draw, (x, y + 29), subtitle, size=16, fill=MUTED, max_width=SIZE)
    canvas.paste(rgb_image(rgb), (x, y + 57))
    draw.rectangle((x, y + 57, x + SIZE - 1, y + 57 + SIZE - 1), outline=BORDER)


def main_canvas(source, edited, shift, name, generic=False):
    width, height = 2 * MARGIN + 2 * SIZE + GAP, 632
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (MARGIN, 18), "One image, prescribed motion", size=28, bold=True)
    direction = "right" if shift > 0 else "left" if shift < 0 else "from the start"
    instruction = f"Requested position: {abs(shift)} pixels {direction}"
    text(draw, (MARGIN, 59), f"{name}  |  {instruction}", size=18, fill=MUTED,
         max_width=width - 2 * MARGIN)
    panel(canvas, (MARGIN, 105), source, "INPUT IMAGE", "The only image supplied to the edit")
    panel(canvas, (MARGIN + SIZE + GAP, 105), edited, "EDITED JEPA READOUT",
          "Coarse 24 x 24 diagnostic colours")
    text(draw, (MARGIN, 568), "The right panel is not a generated full-resolution video.", size=19, bold=True,
         max_width=width - 2 * MARGIN)
    footnote = ("Synthetic-trained probe; readouts of real photographs are unvalidated." if generic else
                "The path is prescribed. No extra frames reveal the hidden background.")
    text(draw, (MARGIN, 598), footnote,
         size=17, fill=MUTED, max_width=width - 2 * MARGIN)
    return canvas


def evidence_canvas(source_probe, edited, target_probe, target_rgb, shift, name):
    width, height = 2 * MARGIN + 2 * SIZE + GAP, 1110
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (MARGIN, 18), "Does the feature edit match the intended image?", size=25, bold=True,
         max_width=width - 2 * MARGIN)
    text(draw, (MARGIN, 56), f"{name}  |  prescribed horizontal shift {shift:+d} px", size=18, fill=MUTED)
    panel(canvas, (MARGIN, 102), source_probe, "ORIGINAL READOUT", "Unedited image features, same probe")
    panel(canvas, (MARGIN + SIZE + GAP, 102), edited, "EDITED READOUT", "Copied features + spatial hole fill")
    panel(canvas, (MARGIN, 571), target_probe, "TARGET READOUT", "True target encoded; evaluation only")
    panel(canvas, (MARGIN + SIZE + GAP, 571), target_rgb, "TRUE TARGET IMAGE", "Procedural ground truth; not an output")
    text(draw, (MARGIN, 1043), "Readouts are 24 x 24 grids enlarged without smoothing.", size=19, bold=True,
         max_width=width - 2 * MARGIN)
    text(draw, (MARGIN, 1075), "Target pixels and features are used for evaluation, never as edit inputs.",
         size=17, fill=MUTED, max_width=width - 2 * MARGIN)
    return canvas


def selection_canvas(source, mask, name):
    width, height = 2 * MARGIN + 2 * SIZE + GAP, 614
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (MARGIN, 18), "One supplied object selection", size=28, bold=True)
    text(draw, (MARGIN, 58), name, size=18, fill=MUTED)
    annotation = source.copy()
    # Mark exactly the selected mask boundary; no detector or new selection.
    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = mask & ~interior
    annotation[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    panel(canvas, (MARGIN, 99), annotation, "SELECTED OBJECT", "White outline is the supplied selection")
    panel(canvas, (MARGIN + SIZE + GAP, 99), mask_rgb, "INPUT MASK", "White pixels identify what to move")
    text(draw, (MARGIN, 566), "The mask is supplied once. Object detection is not being tested.",
         size=18, bold=True, max_width=width - 2 * MARGIN)
    return canvas


def write_mp4(frames, path: Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    width, height = frames[0].size
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "8", "-i", "-",
               "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt",
               "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.size != (width, height) or frame.mode != "RGB":
                raise ValueError("Video canvases must have identical RGB dimensions")
            for _ in range(2):
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stderr.close()
    # Decode every frame, not just the container header.
    subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for completed-video verification")
    metadata = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration",
        "-of", "json", str(path)]))["streams"][0]
    if (int(metadata["nb_read_frames"]) != 2 * len(frames)
            or (metadata["width"], metadata["height"]) != (width, height)):
        raise ValueError("Completed MP4 differs from expected canvas stream")
    return {"format": "mp4", "decode_verified": True, **metadata}


def write_gif(frames, path: Path) -> dict:
    # One shared palette for every frame prevents per-frame palette flicker.
    reference = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for index, frame in enumerate(frames):
        reference.paste(frame, (0, index * frame.height))
    palette = reference.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                 dither=Image.Dither.NONE)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=250,
                    loop=0, optimize=False, disposal=2)
    with Image.open(path) as check:
        count, durations = 0, []
        while True:
            check.load()
            durations.append(check.info.get("duration", 0))
            count += 1
            try:
                check.seek(count)
            except EOFError:
                break
        if count != len(frames) or durations != [250] * len(frames):
            raise ValueError("GIF does not preserve the prescribed display timing")
    return {"format": "gif", "frames": count, "duration_seconds": sum(durations) / 1000,
            "global_palette": True, "dithering": False, "decode_verified": True}


def install_complete(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    descriptor, temporary = tempfile.mkstemp(prefix=".complete-", suffix=destination.suffix,
                                            dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if sha256(destination) != digest:
        raise ValueError("Installed artifact differs from completed staged file")
    return {"file": destination.name, "bytes": destination.stat().st_size, "sha256": digest}


def path_from_record(run: Path, record: dict, key: str, default: Path) -> Path:
    value = record.get(key)
    if value is None and isinstance(record.get("paths"), dict):
        value = record["paths"].get(key)
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else run / path


def load_scene(run: Path, record: dict):
    spec = record["spec"]
    name = spec["name"]
    source_path = path_from_record(run, record, "source_rgb_path", run / "inputs" / name / "source.png")
    mask_path = path_from_record(run, record, "selected_mask_path", run / "inputs" / name / "selected_mask.png")
    prediction_path = path_from_record(run, record, "predictions_path", run / "predictions" / f"{name}.npz")
    source = np.array(Image.open(source_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    if source.shape != (384, 384, 3) or mask.shape != (384, 384) or not mask.any():
        raise ValueError("Invalid saved input image or supplied mask")
    with np.load(prediction_path, allow_pickle=False) as saved:
        shifts = saved["shifts_px"].astype(int).tolist()
        indices = (saved["display_indices"].astype(int).tolist() if "display_indices" in saved
                   else spec["display_indices"] if record.get("generic") else None)
        if shifts != spec["shifts_px"] or indices != spec["display_indices"]:
            raise ValueError("Manifest and saved prediction trajectory differ")
        predictions = {}
        for arm in ARMS:
            key = arm + "__rgb"
            if key not in saved and arm in ("genuine_target", "wrong_direction"):
                continue  # Generic-image mode has no rendered target/reference.
            values = saved[key].copy()
            if values.shape != (len(shifts), 1, 24, 24, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved RGB probe output: {key}")
            predictions[arm] = values[:, 0]
    targets, target_inputs = [], []
    if "genuine_target" in predictions:
        paths = record.get("target_rgb_paths", record.get("paths", {}).get("target_rgb_paths"))
        for index, shift in enumerate(shifts):
            target_path = (Path(paths[index]) if paths is not None else
                           Path("targets") / name / f"dx{shift:+03d}.png")
            if not target_path.is_absolute():
                target_path = run / target_path
            target = np.array(Image.open(target_path).convert("RGB"))
            if target.shape != source.shape:
                raise ValueError("Target image shape differs from source")
            targets.append(target)
            target_inputs.append({"path": str(target_path), "sha256": sha256(target_path)})
    inputs = [{"path": str(path), "sha256": sha256(path)}
              for path in (source_path, mask_path, prediction_path)]
    for item in inputs + target_inputs:
        relative = str(Path(item["path"]).relative_to(run))
        expected = record.get("file_sha256", {}).get(relative)
        if expected is not None and expected != item["sha256"]:
            raise ValueError("Saved display input differs from experiment manifest: " + relative)
    return spec, source, mask, predictions, targets, inputs + target_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed Day 10 experiment directory")
    parser.add_argument("--out", type=Path, help="Default: RUN/analysis")
    args = parser.parse_args()
    run = args.run.resolve()
    out = (args.out or run / "analysis").resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    generic = manifest.get("mode") == "user_image_unscored"
    if generic:
        if not manifest.get("probe_sha256"):
            raise ValueError("Image mode must be run with --probe before its readout can be displayed")
        shifts = manifest["shifts_px"]
        indices = list(range(len(shifts))) + list(range(len(shifts) - 2, -1, -1))
        scenes = [{"spec": {"name": "user_image", "shifts_px": shifts, "display_indices": indices},
                   "source_rgb_path": "source.png", "selected_mask_path": "selected_mask.png",
                   "predictions_path": "predictions.npz", "generic": True}]
    else:
        if not (run / "summary.json").is_file():
            raise ValueError("Visualization requires the completed benchmark summary")
        scenes = manifest["scenes"]
    if not scenes:
        raise ValueError("No completed scenes")
    # The preregistered synthetic test must display all four fixed inputs.
    seeds = tuple(record["spec"].get("seed") for record in scenes)
    if any(seed in EXPECTED_SEEDS for seed in seeds) and seeds != EXPECTED_SEEDS:
        raise ValueError("Fixed test previews require all four preregistered scenes in order")
    report = {"schema": "day10_visualization_v1", "source_manifest_sha256": sha256(manifest_path),
              "scene_selection": "all manifest scenes, in manifest order", "scenes": [],
              "policies": {"probe_grid": [24, 24], "panel_pixels": [384, 384],
                           "resize": "nearest neighbour; no interpolation",
                           "trajectory": "prescribed offsets; independent edits of original source features",
                           "gif_step_ms": 250, "mp4_fps": 8, "mp4_repeats_per_step": 2,
                           "still_index": "maximum prescribed absolute displacement",
                           "readout": "frozen diagnostic RGB probe; not full-resolution video generation",
                           "target_inputs": "evidence panels only; never editor inputs",
                           "staging": "complete and fully decode in /tmp, then atomically install"}}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vjepa-day10-visuals-", dir="/tmp") as temporary:
        stage = Path(temporary)
        for record in scenes:
            spec, source, mask, predictions, targets, inputs = load_scene(run, record)
            name, shifts, indices = spec["name"], spec["shifts_px"], spec["display_indices"]
            if seeds == EXPECTED_SEEDS and tuple(indices) != DISPLAY_INDICES:
                raise ValueError("Unexpected fixed synthetic display path")
            item = {"name": name, "inputs": inputs, "shifts_px": shifts,
                    "display_indices": indices, "artifacts": []}
            preview = [main_canvas(source, predictions["copy_repair"][i], shifts[i], name, generic) for i in indices]
            outputs = [(name + "_preview.mp4", preview, "mp4"),
                       (name + "_preview.gif", preview, "gif")]
            max_index = int(np.argmax(np.abs(shifts)))
            main_still = main_canvas(source, predictions["copy_repair"][max_index], shifts[max_index], name, generic)
            stills = [(name + "_preview.png", main_still),
                      (name + "_selection.png", selection_canvas(source, mask, name))]
            if targets:
                evidence = [evidence_canvas(predictions["noop"][i], predictions["copy_repair"][i],
                                             predictions["genuine_target"][i], targets[i], shifts[i], name)
                            for i in indices]
                outputs.append((name + "_evidence.mp4", evidence, "mp4"))
                stills.append((name + "_evidence.png", evidence_canvas(
                    predictions["noop"][max_index], predictions["copy_repair"][max_index],
                    predictions["genuine_target"][max_index], targets[max_index], shifts[max_index], name)))
            for filename, frames, kind in outputs:
                staged = stage / filename
                metadata = write_mp4(frames, staged) if kind == "mp4" else write_gif(frames, staged)
                item["artifacts"].append({**install_complete(staged, out / filename), **metadata})
            for filename, picture in stills:
                staged = stage / filename
                picture.save(staged)
                with Image.open(staged) as check:
                    check.load()
                    if check.size != picture.size:
                        raise ValueError("Saved PNG dimensions differ from canvas")
                item["artifacts"].append({**install_complete(staged, out / filename),
                                          "format": "png", "resolution": list(picture.size),
                                          "decode_verified": True})
            report["scenes"].append(item)
            print(json.dumps({"scene": name, "artifacts": len(item["artifacts"])}), flush=True)
        staged_report = stage / "visualization_manifest.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        install_complete(staged_report, out / staged_report.name)
    print(json.dumps({"out": str(out), "scenes": len(scenes), "complete": True}), flush=True)


if __name__ == "__main__":
    main()
