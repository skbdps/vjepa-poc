"""Shared-plane classical mask propagation, using frame-0 annotations only.

Parts on one approximately rigid plane share a transform, so a textureless
window can benefit from door/body corners. No corners are initialized on a
later frame: LK correspondences and SIFT reacquisition both retain immutable
frame-0 coordinates. This is a planar-motion baseline, not semantic tracking.

Defaults were fixed before scoring this implementation on sparse labels.
Invalid fits return empty masks for that frame; they never hold stale masks.
The caller must group only parts plausibly belonging to the same plane.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    max_corners: int = 400
    quality_level: float = 0.005
    min_distance: float = 3.0
    context_fraction: float = 0.15
    context_max_pixels: int = 32
    lk_window: int = 21
    lk_levels: int = 3
    fb_max_pixels: float = 1.5
    ransac_pixels: float = 3.0
    min_homography_inliers: int = 6
    min_affine_inliers: int = 3
    min_inlier_fraction: float = 0.5
    sift_ratio: float = 0.75
    reacquire_below: int = 8
    max_normalized_condition: float = 10000.0
    min_area_ratio: float = 0.05
    max_area_ratio: float = 20.0
    min_masked_zncc: float = 0.1


DEFAULT_CONFIG = Config()


def _paths(frames_dir):
    paths = list(Path(frames_dir).glob("*.jpg"))
    if not paths:
        raise ValueError(f"No JPEG frames in {frames_dir}")
    return sorted(paths, key=lambda p: int(p.stem))


def _context_mask(union, cfg):
    yy, xx = np.where(union)
    radius = max(1, min(cfg.context_max_pixels,
                        round(cfg.context_fraction * min(np.ptp(xx) + 1, np.ptp(yy) + 1))))
    return cv2.dilate(union.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))


def _transform_ok(matrix, union, cfg):
    """Reject singular, reflected, unbounded, or extremely distorted warps."""
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if abs(matrix[2, 2]) < 1e-10:
        return False
    matrix = matrix / matrix[2, 2]
    h, w = union.shape
    normalizer = np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1.]])
    normalized = normalizer @ matrix @ np.linalg.inv(normalizer)
    if not np.isfinite(np.linalg.cond(normalized)) or np.linalg.cond(normalized) > cfg.max_normalized_condition:
        return False
    yy, xx = np.where(union)
    box = np.array([[xx.min(), yy.min()], [xx.max(), yy.min()],
                    [xx.max(), yy.max()], [xx.min(), yy.max()]], np.float32)
    denominator = np.c_[box, np.ones(4)] @ matrix[2]
    if np.any(denominator <= 1e-6):
        return False
    projected = cv2.perspectiveTransform(box[None], matrix)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return False
    area0 = cv2.contourArea(box, oriented=True)
    area1 = cv2.contourArea(projected, oriented=True)
    if area0 <= 0 or not cfg.min_area_ratio <= area1 / area0 <= cfg.max_area_ratio:
        return False
    # At least some overlap is needed to emit a visible mask. No stale hold.
    return bool(projected[:, 0].max() >= 0 and projected[:, 0].min() < w
                and projected[:, 1].max() >= 0 and projected[:, 1].min() < h)


def _photo_check(first, current, union, matrix, cfg):
    """Masked zero-mean normalized correlation tolerates global brightness gain.

    The mask is the warped union of the original part prompts, not a later
    annotation. Flat regions cannot provide a correlation check; this fact is
    reported explicitly and geometric checks remain mandatory.
    """
    size = (current.shape[1], current.shape[0])
    mask = cv2.warpPerspective(union.astype(np.uint8), matrix, size,
                               flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    if mask.sum() < 32:
        return False, None, "too_few_visible_pixels"
    template = cv2.warpPerspective(first, matrix, size, flags=cv2.INTER_LINEAR)
    a, b = template[mask].astype(np.float64), current[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    if min(a.std(), b.std()) < 3.0:
        # Strongly textured reference becoming flat is a photometric failure.
        if a.std() >= 3.0 and b.std() < 3.0:
            return False, None, "textured_reference_became_flat"
        return True, None, "uninformative_flat_template"
    correlation = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return correlation >= cfg.min_masked_zncc, correlation, "masked_zncc"


def _fit_candidates(original, moved, cfg):
    """Homography first; full affine fallback handles scarce correspondences."""
    n = len(original)
    if n >= cfg.min_homography_inliers:
        matrix, inliers = cv2.findHomography(original, moved, cv2.RANSAC,
                                             cfg.ransac_pixels, maxIters=2000, confidence=0.995)
        if matrix is not None and inliers is not None:
            good = inliers.ravel().astype(bool)
            if good.sum() >= cfg.min_homography_inliers and good.mean() >= cfg.min_inlier_fraction:
                yield "homography", matrix, good
    if n >= cfg.min_affine_inliers:
        affine, inliers = cv2.estimateAffine2D(original, moved, method=cv2.RANSAC,
                                               ransacReprojThreshold=cfg.ransac_pixels,
                                               maxIters=2000, confidence=0.99, refineIters=10)
        if affine is not None and inliers is not None:
            good = inliers.ravel().astype(bool)
            if good.sum() >= cfg.min_affine_inliers and good.mean() >= cfg.min_inlier_fraction:
                yield "full_affine", np.vstack([affine, [0., 0., 1.]]), good


def _lk(previous, current, original, points, cfg):
    if len(points) == 0:
        return original, points
    kwargs = dict(winSize=(cfg.lk_window,) * 2, maxLevel=cfg.lk_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points[:, None], None, **kwargs)
    if moved is None or ok is None:
        return original[:0], points[:0]
    # Avoid sending nonfinite positions back into native OpenCV code.
    valid = ok.ravel().astype(bool) & np.isfinite(moved[:, 0]).all(axis=1)
    source, dest, anchors = points[valid], moved[valid, 0], original[valid]
    if not len(dest):
        return anchors, dest
    back, okback, _ = cv2.calcOpticalFlowPyrLK(current, previous, dest[:, None], None, **kwargs)
    if back is None or okback is None:
        return anchors[:0], dest[:0]
    valid = okback.ravel().astype(bool) & np.isfinite(back[:, 0]).all(axis=1)
    valid &= np.linalg.norm(back[:, 0] - source, axis=1) <= cfg.fb_max_pixels
    h, w = current.shape
    valid &= (dest[:, 0] >= 0) & (dest[:, 0] < w) & (dest[:, 1] >= 0) & (dest[:, 1] < h)
    return anchors[valid], dest[valid]


def _sift_matches(sift, initial_keypoints, initial_descriptors, current, cfg):
    empty = np.empty((0, 2), np.float32)
    if initial_descriptors is None or len(initial_descriptors) < 2:
        return empty, empty
    keypoints, descriptors = sift.detectAndCompute(current, None)
    if descriptors is None or len(descriptors) < 2:
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(initial_descriptors, descriptors, k=2)
    reverse = matcher.match(descriptors, initial_descriptors)
    reverse_best = {m.queryIdx: m.trainIdx for m in reverse}
    matches = [pair[0] for pair in forward if len(pair) == 2
               and pair[0].distance < cfg.sift_ratio * pair[1].distance
               and reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx]
    if not matches:
        return empty, empty
    return (np.array([initial_keypoints[m.queryIdx].pt for m in matches], np.float32),
            np.array([keypoints[m.trainIdx].pt for m in matches], np.float32))


def run_shared_homography(frames_dir, initial_masks, config=DEFAULT_CONFIG):
    """Return bool masks[T,K,H,W], sorted IDs, and JSON-safe diagnostics.

    Inputs contain only frame-0 masks and raw video frames. Failed fits mark
    every part absent in that frame; immutable SIFT features may reacquire on
    subsequent frames. Identical global transform is applied to all parts.
    """
    start = time.perf_counter()
    cfg = config
    paths = _paths(frames_dir)
    ids = sorted(initial_masks)
    if not ids:
        raise ValueError("Supply at least one frame-0 mask")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"Unreadable image: {paths[0]}")
    masks0 = np.array([np.asarray(initial_masks[i], bool) for i in ids])
    if masks0.shape != (len(ids), *first.shape) or not masks0.reshape(len(ids), -1).any(axis=1).all():
        raise ValueError("Initial masks must be nonempty and match frame dimensions")
    union = masks0.any(axis=0)
    context = _context_mask(union, cfg)
    points = cv2.goodFeaturesToTrack(first, maxCorners=cfg.max_corners,
                                     qualityLevel=cfg.quality_level, minDistance=cfg.min_distance,
                                     mask=context, blockSize=3)
    points = np.empty((0, 2), np.float32) if points is None else points[:, 0]
    original = points.copy()
    # These descriptors are computed once, from frame 0; never updated.
    sift = cv2.SIFT_create(nfeatures=2000)
    keypoints0, descriptors0 = sift.detectAndCompute(first, context)
    output = np.zeros((len(paths), len(ids), *first.shape), bool)
    output[0] = masks0
    rows = [{"frame": 0, "status": "initial_prompt", "points": len(points),
             "matrix": np.eye(3).tolist()}]
    previous = first
    for t, path in enumerate(paths[1:], 1):
        current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if current is None or current.shape != first.shape:
            raise ValueError(f"Unreadable or inconsistent frame: {path}")
        original, points = _lk(previous, current, original, points, cfg)
        row = {"frame": t, "status": "absent_fit_failed", "lk_points": len(points), "attempts": []}
        accepted = None

        def consider(anchors, locations, source):
            for model, matrix, inliers in _fit_candidates(anchors, locations, cfg):
                record = {"source": source, "model": model, "inliers": int(inliers.sum()),
                          "correspondences": len(anchors), "geometry_ok": bool(_transform_ok(matrix, union, cfg))}
                row["attempts"].append(record)
                if not record["geometry_ok"]:
                    continue
                passed, correlation, reason = _photo_check(first, current, union, matrix, cfg)
                record.update(photo_ok=bool(passed), masked_zncc=correlation, photo_reason=reason)
                if passed:
                    return matrix, anchors[inliers], locations[inliers], source, model
            return None

        if len(points) >= cfg.reacquire_below:
            accepted = consider(original, points, "lk")
        if accepted is None:
            sift_original, sift_points = _sift_matches(sift, keypoints0, descriptors0, current, cfg)
            row["sift_matches"] = len(sift_points)
            accepted = consider(sift_original, sift_points, "initial_sift")
        if accepted is None and len(points) < cfg.reacquire_below:
            accepted = consider(original, points, "lk")
        if accepted is not None:
            matrix, original, points, source, model = accepted
            for j, mask in enumerate(masks0):
                output[t, j] = cv2.warpPerspective(mask.astype(np.uint8), matrix,
                                                  (first.shape[1], first.shape[0]),
                                                  flags=cv2.INTER_NEAREST).astype(bool)
            row.update(status="tracked", source=source, model=model,
                       points=len(points), matrix=(matrix / matrix[2, 2]).tolist())
        rows.append(row)
        previous = current
    diagnostics = {"method": "shared_plane_lk_sift_homography", "config": asdict(cfg),
                   "prompt_frames": [0], "shared_transform": True,
                   "future_labels_used": False, "opencv": cv2.__version__,
                   "initial_lk_points": rows[0]["points"], "initial_sift_points": len(keypoints0),
                   "failure_policy": "empty masks; no stale hold", "frames": rows,
                   "seconds": time.perf_counter() - start}
    return output, ids, diagnostics


def self_test():
    """Geometry, mask transport, blank-frame failure, and reacquisition check."""
    import tempfile
    rng = np.random.default_rng(405)
    first = rng.integers(0, 256, (160, 220), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (3, 3), 0.5)
    initial = {1: np.zeros(first.shape, bool), 2: np.zeros(first.shape, bool)}
    initial[1][45:100, 65:130] = True
    initial[2][25:45, 70:125] = True
    union = initial[1] | initial[2]
    assert not _transform_ok(np.full((3, 3), np.nan), union, DEFAULT_CONFIG)
    assert not _transform_ok(np.zeros((3, 3)), union, DEFAULT_CONFIG)
    assert not _transform_ok(np.diag([-1., 1., 1.]), union, DEFAULT_CONFIG)
    with tempfile.TemporaryDirectory() as folder:
        transforms = [np.array([[1., 0, 3 * t], [0, 1., 2 * t], [0, 0, 1.]]) for t in range(5)]
        for t, matrix in enumerate(transforms):
            im = cv2.warpPerspective(first, matrix, (220, 160))
            if t == 3:
                im[:] = 0
            cv2.imwrite(str(Path(folder) / f"{t:05d}.jpg"), im)
        masks, ids, diagnostics = run_shared_homography(folder, initial)
    minimum_iou = 1.0
    for t in (1, 2, 4):
        for j, identity in enumerate(ids):
            expected = cv2.warpPerspective(initial[identity].astype(np.uint8), transforms[t],
                                            (220, 160), flags=cv2.INTER_NEAREST).astype(bool)
            iou = float((expected & masks[t, j]).sum() / (expected | masks[t, j]).sum())
            minimum_iou = min(minimum_iou, iou)
            assert iou > 0.9, (t, identity, iou)
    assert not masks[3].any(), "Blank frame must not emit a stale mask"
    assert diagnostics["frames"][4]["source"] == "initial_sift"
    return {"passed": True, "minimum_synthetic_iou": minimum_iou,
            "blank_frame_empty": True, "reacquired_from_initial_template": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
