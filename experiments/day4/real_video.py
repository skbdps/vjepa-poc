"""Real-video diagnostic: initialize named car parts once, propagate, then edit.

This is an exploratory demonstration on DAVIS footage, not a new benchmark.
Sparse polygon annotations are created independently of model predictions.
Only annotation frame 0 is supplied to trackers. Later annotations only score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

SAM2_REPOSITORY = "https://github.com/facebookresearch/sam2.git"
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DAVIS_FRAME_BASE = "https://graphics.ethz.ch/Downloads/Data/Davis/files/sequences"
DAVIS_SOURCE_PAGE = "https://davischallenge.org/davis2016/browse.html"
DAVIS_FRAME_INDEX = "https://davischallenge.org/json_data/global.js"


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_paths(frames_dir):
    paths = sorted(Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        raise ValueError(f"No JPEG frames in {frames_dir}")
    return paths


def download_frames(outfolder, sequence="car-roundabout", max_frames=64, start=0,
                    stride=1, workers=4):
    """Download a bounded contiguous/subsampled sequence from official DAVIS.

    Names are renumbered for SAM2; source frame indices and file hashes persist.
    A 404 is an error, never silently interpreted as a complete sequence.
    """
    if sequence not in {"car-roundabout", "car-shadow", "car-turn"}:
        raise ValueError("This diagnostic only supports the three reviewed car sequences")
    if max_frames <= 0 or stride <= 0 or start < 0:
        raise ValueError("Invalid frame selection")
    outfolder = Path(outfolder)
    frames_dir = outfolder / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = outfolder / "source_manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text())
        if any(previous.get(k) != v for k, v in {
                "sequence": sequence, "start": start, "stride": stride,
                "frame_count": max_frames}.items()):
            raise ValueError("Output folder already records a different frame selection")
    # Prevent old frames silently changing the sequence length on a new selection.
    expected = {f"{i:05d}.jpg" for i in range(max_frames)}
    if {p.name for p in frames_dir.glob("*.jpg")} - expected:
        raise ValueError("Output frames directory contains files outside this selection")

    def one(i):
        source_index = start + i * stride
        url = f"{DAVIS_FRAME_BASE}/{sequence}/{source_index:05d}.jpg"
        path = frames_dir / f"{i:05d}.jpg"
        if not path.exists():
            last_error = None
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=90) as response:
                        data = response.read()
                    temp = path.with_suffix(".part")
                    temp.write_bytes(data)
                    with Image.open(temp) as im:
                        im.verify()
                    temp.replace(path)
                    break
                except Exception as e:
                    last_error = e
                    if attempt == 2:
                        raise RuntimeError(f"Failed {url}") from last_error
        with Image.open(path) as im:
            size = list(im.size)
        return {"frame_index": i, "source_index": source_index, "url": url,
                "sha256": sha256_file(path), "size_wh": size}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, range(max_frames)))
    if len({tuple(row["size_wh"]) for row in rows}) != 1:
        raise ValueError("Frames have inconsistent sizes")
    manifest = {"dataset": "DAVIS", "sequence": sequence,
                "source_page": DAVIS_SOURCE_PAGE, "source_index_page": DAVIS_FRAME_INDEX,
                "frames_dir": str(frames_dir), "frame_count": len(rows),
                "start": start, "stride": stride, "frames": rows,
                "licensing_note": "See ATTRIBUTION.md; footage rights are separate from SAM2 code/weights.",
                "benchmark_note": "Exploratory part labels; these are not official DAVIS object annotations."}
    write_json(outfolder / "source_manifest.json", manifest)
    contact_sheet(frames_dir, outfolder / "contact_sheet.jpg")
    return manifest


def contact_sheet(frames_dir, output, indices=None, width=427):
    paths = frame_paths(frames_dir)
    if indices is None:
        indices = np.unique(np.linspace(0, len(paths) - 1, 12).round().astype(int)).tolist()
    im0 = Image.open(paths[0])
    height = round(width * im0.height / im0.width)
    canvas = Image.new("RGB", (width * 3, (height + 26) * ((len(indices) + 2) // 3)), "white")
    draw = ImageDraw.Draw(canvas)
    for n, idx in enumerate(indices):
        im = Image.open(paths[idx]).convert("RGB").resize((width, height))
        x, y = (n % 3) * width, (n // 3) * (height + 26)
        canvas.paste(im, (x, y + 26))
        draw.text((x + 8, y + 5), f"Frame {idx}", fill="black")
    canvas.save(output, quality=94)


def prepare_sam2(root="/content/sam2", checkpoint_dir="/content/checkpoints"):
    """Install pinned official code with optional CUDA extension disabled."""
    root, checkpoint_dir = Path(root), Path(checkpoint_dir)
    if not (root / ".git").exists():
        subprocess.run(["git", "clone", SAM2_REPOSITORY, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", SAM2_REVISION], check=True)
    env = dict(os.environ, SAM2_BUILD_CUDA="0")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(root)],
                   env=env, check=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "sam2.1_hiera_tiny.pt"
    if not checkpoint.exists():
        with urlopen(SAM2_CHECKPOINT, timeout=180) as response, checkpoint.with_suffix(".part").open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        checkpoint.with_suffix(".part").replace(checkpoint)
    # Editable installs created inside a live notebook need this import path now.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return str(checkpoint)


def polygon_mask(item, shape_hw):
    mask = np.zeros(shape_hw, np.uint8)
    if not item.get("visible", True):
        return mask.astype(bool)
    polygons = item.get("polygons", [item.get("polygon", [])])
    for polygon in polygons:
        pts = np.asarray(polygon, np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            raise ValueError("Visible annotation needs a polygon of >=3 xy points")
        if (pts[:, 0] < 0).any() or (pts[:, 0] >= shape_hw[1]).any() or (pts[:, 1] < 0).any() or (pts[:, 1] >= shape_hw[0]).any():
            raise ValueError("Polygon outside image dimensions")
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def initial_masks(annotations, shape_hw):
    """Explicit information barrier: return frame 0 masks only."""
    labels = annotations["frames"]["0"]
    masks = {int(part["id"]): polygon_mask(labels[str(part["id"])], shape_hw)
             for part in annotations["parts"]}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("All tagged parts must be visible and annotated in frame 0")
    return masks


def run_sam2(frames_dir, initial_mask, outfolder, checkpoint=None, predictor=None):
    """Track masks with SAM2.1tiny; receives no later-frame annotations.

    T4 lacks native BF16, so use FP32 for this compact model for reproducibility.
    GPU video images are offloaded to CPU; mask memory remains on GPU.
    """
    import torch
    outfolder = Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 GPU experiment requires CUDA")
    if predictor is None:
        from sam2.build_sam import build_sam2_video_predictor
        if checkpoint is None:
            raise ValueError("Provide checkpoint or predictor")
        predictor = build_sam2_video_predictor(SAM2_CONFIG, checkpoint, device="cuda",
                                               apply_postprocessing=False)
    paths = frame_paths(frames_dir)
    ids = sorted(initial_mask)
    shape = next(iter(initial_mask.values())).shape
    masks = np.zeros((len(paths), len(ids), *shape), dtype=bool)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=True)
        for obj_id in ids:
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=obj_id,
                                   mask=initial_mask[obj_id])
        received = set()
        for index, object_ids, logits in predictor.propagate_in_video(state):
            if not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite SAM2 logits in frame {index}")
            for j, obj_id in enumerate(object_ids):
                masks[index, ids.index(obj_id)] = (logits[j, 0] > 0).cpu().numpy()
            received.add(index)
        if received != set(range(len(paths))):
            raise ValueError("SAM2 did not return all frames")
        predictor.reset_state(state)
    torch.cuda.synchronize()
    timing = {"seconds": time.perf_counter() - start,
              "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
              "precision": "float32", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "model_revision": SAM2_REVISION,
              "checkpoint": SAM2_CHECKPOINT, "config": SAM2_CONFIG,
              "postprocessing": False, "prompt_frames": [0]}
    if checkpoint:
        timing["checkpoint_sha256"] = sha256_file(checkpoint)
    np.savez_compressed(outfolder / "sam2_masks.npz", ids=np.asarray(ids), masks=masks)
    write_json(outfolder / "sam2_manifest.json", timing)
    return masks, ids, timing


def run_affine_lk(frames_dir, initial_mask):
    """Strong classical baseline: tracked corners + per-part RANSAC affine warp.

    Feature points are selected only from each first-frame part. No reseeding or
    future labels. Fewer than three reliable correspondences marks part absent.
    """
    paths = frame_paths(frames_dir)
    images = [cv2.imread(str(p)) for p in paths]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    ids = sorted(initial_mask)
    shape = grays[0].shape
    output = np.zeros((len(paths), len(ids), *shape), bool)
    for j, obj_id in enumerate(ids):
        mask0 = initial_mask[obj_id].astype(np.uint8)
        output[0, j] = mask0
        points = cv2.goodFeaturesToTrack(grays[0], maxCorners=120, qualityLevel=.005,
                                         minDistance=3, mask=mask0, blockSize=3)
        original = points.copy() if points is not None else None
        for t in range(1, len(paths)):
            if points is None or len(points) < 3:
                break
            moved, valid, _ = cv2.calcOpticalFlowPyrLK(grays[t - 1], grays[t], points, None,
                                                     winSize=(21, 21), maxLevel=3)
            if moved is None or valid is None or not np.isfinite(moved).all():
                break
            back, backward_valid, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t - 1], moved, None,
                                                              winSize=(21, 21), maxLevel=3)
            if back is None or backward_valid is None:
                break
            good = valid[:, 0].astype(bool) & backward_valid[:, 0].astype(bool)
            good &= np.isfinite(back[:, 0]).all(axis=1)
            good &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) <= 1.5
            xy = moved[:, 0]
            good &= (xy[:, 0] >= 0) & (xy[:, 0] < shape[1])
            good &= (xy[:, 1] >= 0) & (xy[:, 1] < shape[0])
            points, original = moved[good], original[good]
            if len(points) < 3:
                break
            matrix, _ = cv2.estimateAffinePartial2D(original[:, 0], points[:, 0],
                                                   method=cv2.RANSAC, ransacReprojThreshold=3)
            if matrix is not None:
                output[t, j] = cv2.warpAffine(mask0, matrix, (shape[1], shape[0]),
                                             flags=cv2.INTER_NEAREST).astype(bool)
    return output, ids


def score_masks(masks, ids, annotations):
    rows = []
    for key, labels in annotations["frames"].items():
        t = int(key)
        if t == 0:
            continue
        if t < 0 or t >= len(masks):
            raise ValueError(f"Annotation frame {t} outside prediction")
        for obj_id in ids:
            if str(obj_id) not in labels:
                continue
            item = labels[str(obj_id)]
            if item.get("ignore", False):
                continue
            target = polygon_mask(item, masks.shape[-2:])
            prediction = masks[t, ids.index(obj_id)]
            overlap = int((target & prediction).sum())
            union = int((target | prediction).sum())
            area, truth_area = int(prediction.sum()), int(target.sum())
            rows.append({"frame": t, "part_id": obj_id, "visible": bool(truth_area),
                         "iou": overlap / union if union else 1.0,
                         "precision": overlap / area if area else (1.0 if not truth_area else 0.0),
                         "recall": overlap / truth_area if truth_area else None,
                         "predicted_pixels": area, "target_pixels": truth_area,
                         "spill_pixels": area - overlap,
                         "false_presence": bool(area) if not truth_area else None})
    if not rows:
        raise ValueError("Need at least one scored annotation after frame 0")
    visible = [row for row in rows if row["visible"]]
    absent = [row for row in rows if not row["visible"]]
    return {"mean_iou": float(np.mean([r["iou"] for r in rows])),
            "visible_mean_iou": float(np.mean([r["iou"] for r in visible])) if visible else None,
            "mean_precision": float(np.mean([r["precision"] for r in rows])),
            "visible_mean_recall": float(np.mean([r["recall"] for r in visible])) if visible else None,
            "scored_part_frames": len(rows), "visible_part_frames": len(visible),
            "absent_part_frames": len(absent),
            "absent_false_presence_count": sum(r["false_presence"] for r in absent), "rows": rows}


def render_comparison(frames_dir, predictions, ids, part_names, output, fps=12):
    """Original / stable tag overlay / one persistent part recolor per method."""
    from part_editor import apply_recolor
    paths = frame_paths(frames_dir)
    image0 = cv2.imread(str(paths[0]))
    h, w = image0.shape[:2]
    row_h = h + 35
    methods = list(predictions)
    colors = [(245, 90, 30), (30, 210, 245), (170, 80, 200), (80, 220, 100)]  # BGR
    raw = Path(output).with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, row_h * len(methods)))
    if not writer.isOpened():
        raise RuntimeError("Could not create video writer")
    for t, path in enumerate(paths):
        original = cv2.imread(str(path))
        rows = []
        for method in methods:
            overlay = original.copy()
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            edited_rgb, _, _ = apply_recolor(rgb, predictions[method][t], ids, ids[0])
            edited = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)
            for j, obj_id in enumerate(ids):
                mask = predictions[method][t, j]
                color = np.asarray(colors[j % len(colors)], dtype=np.float32)
                overlay[mask] = np.rint(.55 * original[mask] + .45 * color).astype(np.uint8)
                if mask.any():
                    yy, xx = np.where(mask)
                    cv2.putText(overlay, f"{obj_id}:{part_names[obj_id]}", (int(xx.mean()), int(yy.mean())),
                                cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            tiles = [original, overlay, edited]
            row = np.zeros((row_h, w * 3, 3), np.uint8)
            row[35:] = np.concatenate(tiles, axis=1)
            for col, label in enumerate(["Original", f"{method}: named part masks", f"{method}: persistent door recolor"]):
                cv2.putText(row, f"{label} | frame {t}", (col * w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)
        writer.write(np.concatenate(rows))
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(output)], check=True)
    raw.unlink()


def run(frames_dir, annotations_path, outfolder, checkpoint=None, predictor=None):
    from part_editor import edit_masks, replay
    from classical_masks import run_shared_homography
    outfolder = Path(outfolder)
    outfolder.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(Path(annotations_path).read_text())
    shape = np.asarray(Image.open(frame_paths(frames_dir)[0])).shape[:2]
    initial = initial_masks(annotations, shape)
    sam_masks, ids, timing = run_sam2(frames_dir, initial, outfolder, checkpoint, predictor)
    flow_masks, flow_ids = run_affine_lk(frames_dir, initial)
    assert ids == flow_ids
    shared_masks, shared_ids, shared_diagnostics = run_shared_homography(frames_dir, initial)
    assert ids == shared_ids
    np.savez_compressed(outfolder / 'shared_homography_masks.npz', ids=np.asarray(ids), masks=shared_masks)
    write_json(outfolder / 'shared_homography_diagnostics.json', shared_diagnostics)
    predictions = {"SAM2.1 tiny": sam_masks, "LK affine": flow_masks, "Shared homography": shared_masks}
    results = {method: score_masks(masks, ids, annotations) for method, masks in predictions.items()}
    results['selective_edit'] = {method: score_masks(edit_masks(masks, ids, ids[0])[:, None], [ids[0]], annotations)
                                 for method, masks in predictions.items()}
    write_json(outfolder / "results.json", results)
    np.savez_compressed(outfolder / "lk_masks.npz", ids=np.asarray(ids), masks=flow_masks)
    registry = {"sequence": annotations["sequence"], "parts": annotations["parts"],
                "initial_annotation_frame": 0, "tracker_correction_frames": [],
                "editing": {"operation": "non-generative recolor", "selected_id": ids[0],
                            "unselected_part_ids": ids[1:], "scope": "same identifier on every predicted mask"},
                "annotation_provenance": "Assistant visually authored approximate polygons; not human ground truth.",
                "limitations": ["Sparse assistant-authored polygons are approximate diagnostic labels.",
                                "No learned video synthesis, generative edit, face identity, or generalization claim.",
                                "SAM2 pretraining may include DAVIS; this is a pipeline demonstration."]}
    write_json(outfolder / "part_registry.json", registry)
    names = {int(p["id"]): p["name"] for p in annotations["parts"]}
    render_comparison(frames_dir, {m: predictions[m] for m in ('SAM2.1 tiny', 'Shared homography')}, ids, names, outfolder / "part_tracking_edit.mp4")
    replay(frames_dir, outfolder / 'sam2_masks.npz', outfolder / 'door_recolor.mp4', target_id=ids[0])
    print(json.dumps({name: {key: value for key, value in result.items() if key != "rows"}
                      for name, result in results.items() if name != 'selective_edit'}, indent=2))
    return results, timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--out", required=True)
    download.add_argument("--sequence", default="car-roundabout")
    download.add_argument("--frames", type=int, default=64)
    infer = sub.add_parser("run")
    infer.add_argument("--frames-dir", required=True)
    infer.add_argument("--annotations", required=True)
    infer.add_argument("--out", required=True)
    infer.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download_frames(args.out, args.sequence, args.frames), indent=2))
    else:
        run(args.frames_dir, args.annotations, args.out, args.checkpoint)
