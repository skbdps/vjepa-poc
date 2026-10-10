"""Persistent part localization from frame-zero annotations only.

The encoder is frozen and may be bidirectional inside its input window. This
module's state updates are causal in *feature steps*, not necessarily in video
frames. A step is a pair of frames; output coordinates are 16-pixel grid cells.

The appearance memory has two slots: an immutable initial prototype and a
conservatively updated prototype. A bounded Lucas--Kanade motion bonus cannot
eliminate global candidates, so recovery does not require the old location.
No function accepts future masks, target trajectories, or evaluation labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SIZE, PATCH, GRID = 384, 16, 24


@dataclass
class Predictions:
    scores: np.ndarray
    cells: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class TrackerConfig:
    # Scores are cosine similarities. Defaults fixed before day4 test access.
    context_weight: float = .15
    motion_weight: float = .035
    memory_weight: float = .20
    update_rate: float = .10
    state_threshold: float = .79  # day3 calibration is development evidence
    update_margin: float = .01
    motion_sigma: float = 2.0  # grid cells
    fb_error: float = 1.5      # pixels


@dataclass(frozen=True)
class TemplateConfig:
    context_px: int = 12
    scales: tuple = (.80, .90, 1., 1.10, 1.20)
    motion_weight: float = .08
    state_threshold: float = .60
    update_margin: float = .02
    motion_sigma: float = 2.0
    fb_error: float = 1.5


def _unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _validate_frames(frames):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != (SIZE, SIZE, 3) or len(frames) % 2:
        raise ValueError(f"Expected [even T,384,384,3], received {frames.shape}")
    return frames


def initial_patch_labels(initial_mask):
    """Frame-zero labels only; use benchmark.initial_labels when available."""
    initial_mask = np.asarray(initial_mask)
    if initial_mask.shape != (SIZE, SIZE):
        raise ValueError("Initial mask must be [384,384]")
    labels = np.zeros((GRID, GRID), np.int16)
    for pid in np.unique(initial_mask):
        if pid <= 0:
            continue
        fraction = (initial_mask == pid).reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
        labels[fraction >= .65] = pid
        if not np.any(labels == pid):
            raise ValueError(f"Part {pid} has no initial patch with >=65% coverage")
    return labels


def _initialization(features, initial_labels):
    features = np.asarray(features, np.float32)
    if features.ndim != 4 or features.shape[1:3] != (GRID, GRID):
        raise ValueError(f"Expected [steps,24,24,D], received {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite")
    labels = np.asarray(initial_labels)
    if labels.shape != (GRID, GRID):
        raise ValueError("Initial labels must be [24,24]")
    ids = np.sort(np.unique(labels[labels > 0]))
    if not len(ids):
        raise ValueError("At least one initial part is required")
    unit = _unit(features)
    prototypes, anchors = [], []
    for pid in ids:
        coordinates = np.argwhere(labels == pid)
        center = coordinates.mean(0)
        anchors.append(coordinates[((coordinates-center)**2).sum(1).argmin()])
        prototypes.append(_unit(unit[0][labels == pid].mean(0)))
    return unit, np.stack(prototypes), np.stack(anchors), ids


def match_scoremaps(features, initial_labels):
    """Frozen initial mean prototypes: return [steps,parts,row,col] cosine."""
    unit, prototypes, _, _ = _initialization(features, initial_labels)
    return np.einsum("tyxd,kd->tkyx", unit, prototypes, optimize=True)


def global_match(features, initial_labels):
    maps = match_scoremaps(features, initial_labels)
    flat = maps.reshape(*maps.shape[:2], -1)
    ix = flat.argmax(-1)
    return Predictions(flat.max(-1), np.stack((ix//GRID, ix % GRID), -1),
                       np.ones(ix.shape, bool))


def _context_maps(unit, anchors):
    """Four fixed relative context samples, shifted with each candidate.

    The samples come from two grid cells away from the initial anchor. This is
    an explicitly translation-only contextual cue, not a shape/pose model.
    Unsupported image-edge offsets are omitted rather than zero-padded.
    """
    steps, height, width, _ = unit.shape
    result = np.zeros((steps, len(anchors), height, width), np.float32)
    for ki, (ar, ac) in enumerate(anchors):
        count = np.zeros((height, width), np.float32)
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            ir, ic = ar+dr, ac+dc
            if not (0 <= ir < height and 0 <= ic < width):
                continue
            r0, r1 = max(0, -dr), min(height, height-dr)
            c0, c1 = max(0, -dc), min(width, width-dc)
            context = np.einsum("tyxd,d->tyx", unit[:, r0+dr:r1+dr, c0+dc:c1+dc],
                                unit[0, ir, ic], optimize=True)
            result[:, ki, r0:r1, c0:c1] += context
            count[r0:r1, c0:c1] += 1
        result[:, ki] /= np.maximum(count, 1)
    return result


def _pair_images(frames):
    frames = _validate_frames(frames)
    # Midpoint appearance is shared by semantic motion and the NCC comparator.
    return np.rint(frames.astype(np.float32).reshape(-1, 2, SIZE, SIZE, 3).mean(1)).astype(np.uint8)


def _pair_greys(frames):
    import cv2
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in _pair_images(frames)]


def _flow_prediction(previous, current, xy, radius, fb_error):
    """Re-seeded local LK with forward/backward and photometric validation."""
    import cv2
    if xy is None:
        return None
    x, y = xy
    mask = np.zeros_like(previous)
    x0, x1 = max(0, int(x-radius)), min(SIZE, int(x+radius+1))
    y0, y1 = max(0, int(y-radius)), min(SIZE, int(y+radius+1))
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=24, qualityLevel=.01,
                                     minDistance=3, mask=mask, blockSize=3)
    if points is None or len(points) < 2:
        return None
    settings = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    nxt, forward, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **settings)
    if nxt is None:
        return None
    back, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None, **settings)
    if back is None:
        return None
    good = forward[:, 0].astype(bool) & backward[:, 0].astype(bool)
    good &= np.isfinite(nxt[:, 0]).all(1) & np.isfinite(back[:, 0]).all(1)
    good &= np.linalg.norm(back[:, 0]-points[:, 0], axis=1) <= fb_error
    good &= error[:, 0] <= 25.
    good &= (nxt[:, 0] >= 0).all(1) & (nxt[:, 0] < SIZE).all(1)
    if good.sum() < 2 or good.mean() < .3:
        return None
    displacement = nxt[good, 0]-points[good, 0]
    median = np.median(displacement, axis=0)
    inliers = np.linalg.norm(displacement-median, axis=1) <= 4.
    if inliers.sum() < 2:
        return None
    return np.asarray(xy)+np.median(displacement[inliers], axis=0)


def _peak_margin(scoremap, row, col):
    yy, xx = np.indices(scoremap.shape)
    other = (yy-row)**2+(xx-col)**2 > 2.5**2
    return float(scoremap[row, col]-np.max(scoremap[other]))


def _motion_bonus(xy, weight, sigma):
    if xy is None or weight <= 0:
        return 0.
    yy, xx = np.indices((GRID, GRID))
    x, y = np.asarray(xy)/PATCH-.5
    return weight*np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))


def track_parts(features, frames, initial_mask, config=TrackerConfig(), initial_labels=None):
    """Global semantic retrieval + validated motion + anchored online memory.

    ``scores`` exclude the motion bonus: being near a prediction is not evidence
    of visibility. Calibrate the final output threshold on calibration clips.
    ``state_threshold`` controls updates only and is fixed independently of test
    labels. Set context/motion/memory weights to zero for controlled ablations.
    """
    frames = _validate_frames(frames)
    labels = initial_patch_labels(initial_mask) if initial_labels is None else initial_labels
    unit, initial, anchors, ids = _initialization(features, labels)
    steps, parts = len(unit), len(ids)
    if len(frames) != 2*steps:
        raise ValueError("Feature and video time dimensions differ")
    maps = np.einsum("tyxd,kd->tkyx", unit, initial, optimize=True)
    if config.context_weight:
        maps = (1-config.context_weight)*maps+config.context_weight*_context_maps(unit, anchors)
    greys = _pair_greys(frames) if config.motion_weight else None
    scores = np.empty((steps, parts), np.float32)
    cells = np.empty((steps, parts, 2), np.int16)
    centers = [anchor[::-1].astype(float)*PATCH+PATCH/2 for anchor in anchors]
    memory = initial.copy()
    for t in range(steps):
        for ki in range(parts):
            appearance = maps[t, ki].copy()
            if config.memory_weight and t:
                adapted = np.einsum("yxd,d->yx", unit[t], memory[ki], optimize=True)
                # Original identity is an immutable anchor. Online memory is
                # supplementary and cannot reduce the initial score.
                appearance += config.memory_weight*np.maximum(0., adapted-appearance)
            predicted = None
            if t and config.motion_weight:
                predicted = _flow_prediction(greys[t-1], greys[t], centers[ki],
                                             26, config.fb_error)
            selection = appearance+_motion_bonus(predicted, config.motion_weight, config.motion_sigma)
            row, col = np.unravel_index(selection.argmax(), selection.shape)
            confidence = float(appearance[row, col])
            scores[t, ki], cells[t, ki] = confidence, (row, col)
            reliable = (confidence >= config.state_threshold and
                        _peak_margin(appearance, row, col) >= config.update_margin)
            centers[ki] = np.array([col+.5, row+.5])*PATCH if reliable else None
            if t and reliable and config.memory_weight:
                # Only a location supported by the immutable appearance model
                # can enter memory, preventing self-confirming memory updates.
                if maps[t, ki, row, col] >= config.state_threshold:
                    memory[ki] = _unit((1-config.update_rate)*memory[ki] +
                                       config.update_rate*unit[t, row, col])
    return Predictions(scores, cells, np.ones((steps, parts), bool))


def _template_scoremap(image, template, scales):
    """NCC sampled at exactly the same patch centers as semantic predictions."""
    import cv2
    best = np.full((GRID, GRID), -1., np.float32)
    for scale in scales:
        h, w = np.maximum(3, np.rint(np.asarray(template.shape[:2])*scale).astype(int))
        if h >= SIZE or w >= SIZE:
            continue
        resized = cv2.resize(template, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        corr = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        # Each output is a location cell. Within its 16x16 square, allow the
        # template center to vary, avoiding an unfair quantization penalty.
        for row in range(GRID):
            y0, y1 = max(0, row*PATCH-h//2), min(corr.shape[0], (row+1)*PATCH-h//2)
            if y1 <= y0:
                continue
            for col in range(GRID):
                x0, x1 = max(0, col*PATCH-w//2), min(corr.shape[1], (col+1)*PATCH-w//2)
                if x1 > x0:
                    best[row, col] = max(best[row, col], float(corr[y0:y1, x0:x1].max()))
    return np.nan_to_num(best, nan=-1.)


def template_tracker(frames, initial_mask, config=TemplateConfig()):
    """Strong classical comparator: multiscale initial-template NCC + FB-LK.

    Uses immutable RGB templates with surrounding context and global
    retrieval every step, so it can recover after complete occlusion. It has
    the same validated motion bonus as the semantic tracker; no learned model.
    """
    import cv2
    frames = _validate_frames(frames)
    mask = np.asarray(initial_mask)
    if mask.shape != (SIZE, SIZE):
        raise ValueError("Initial mask must be [384,384]")
    ids = np.sort(np.unique(mask[mask > 0]))
    if not len(ids):
        raise ValueError("At least one initial part is required")
    images = _pair_images(frames)
    greys = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in images]
    templates, centers = [], []
    for pid in ids:
        yy, xx = np.where(mask == pid)
        # Symmetric crop keeps the template's center at the target center.
        cx, cy = (xx.min()+xx.max())//2, (yy.min()+yy.max())//2
        rx = min((xx.max()-xx.min())//2+config.context_px, cx, SIZE-1-cx)
        ry = min((yy.max()-yy.min())//2+config.context_px, cy, SIZE-1-cy)
        templates.append(frames[0, cy-ry:cy+ry+1, cx-rx:cx+rx+1])
        centers.append(np.array([cx, cy], float))
    steps, parts = len(greys), len(ids)
    scores = np.empty((steps, parts), np.float32)
    cells = np.empty((steps, parts, 2), np.int16)
    for t, grey in enumerate(greys):
        for ki, template in enumerate(templates):
            appearance = _template_scoremap(images[t], template, config.scales)
            predicted = (_flow_prediction(greys[t-1], grey, centers[ki], 26,
                                         config.fb_error) if t and config.motion_weight else None)
            selection = appearance+_motion_bonus(predicted, config.motion_weight, config.motion_sigma)
            row, col = np.unravel_index(selection.argmax(), selection.shape)
            scores[t, ki], cells[t, ki] = appearance[row, col], (row, col)
            reliable = (scores[t, ki] >= config.state_threshold and
                        _peak_margin(appearance, row, col) >= config.update_margin)
            centers[ki] = np.array([col+.5, row+.5])*PATCH if reliable else None
    return Predictions(scores, cells, np.ones((steps, parts), bool))


def fixed_position(initial_labels, steps):
    labels = np.asarray(initial_labels)
    anchors = []
    for pid in np.sort(np.unique(labels[labels > 0])):
        candidates = np.argwhere(labels == pid)
        anchors.append(candidates[((candidates-candidates.mean(0))**2).sum(1).argmin()])
    cells = np.broadcast_to(anchors, (steps, len(anchors), 2)).copy()
    return Predictions(np.ones(cells.shape[:2]), cells, np.ones(cells.shape[:2], bool))


def optical_flow(frames, initial_mask):
    """Unmodified day3-style LK baseline generalized to arbitrary even T."""
    import cv2
    frames = _validate_frames(frames)
    ids = np.sort(np.unique(np.asarray(initial_mask)[np.asarray(initial_mask) > 0]))
    grey = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    centers = np.full((len(frames), len(ids), 2), np.nan, np.float32)
    for ki, pid in enumerate(ids):
        region = (initial_mask == pid).astype(np.uint8)*255
        ys, xs = np.where(region)
        center = np.array([xs.mean(), ys.mean()], np.float32)
        points = cv2.goodFeaturesToTrack(grey[0], maxCorners=12, qualityLevel=.01,
                                         minDistance=4, mask=region, blockSize=3)
        if points is None:
            continue
        offsets = center-points[:, 0]
        centers[0, ki] = center
        for t in range(1, len(frames)):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(grey[t-1], grey[t], points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            if nxt is None:
                break
            good = status[:, 0].astype(bool) & np.isfinite(nxt[:, 0]).all(1)
            good &= (nxt[:, 0] >= 0).all(1) & (nxt[:, 0] < SIZE).all(1)
            if not good.any():
                break
            points, offsets = nxt[good], offsets[good]
            centers[t, ki] = np.median(points[:, 0]+offsets, axis=0)
    paired = centers.reshape(-1, 2, len(ids), 2)
    valid = np.isfinite(paired).all(axis=(1, 3))
    middle = np.nan_to_num(paired.mean(1), nan=0)
    cells = (middle[..., ::-1]/PATCH).astype(int)
    valid &= (cells >= 0).all(-1) & (cells < GRID).all(-1)
    return Predictions(np.ones(valid.shape), cells.clip(0, GRID-1), valid)


def self_test():
    """Implementation checks with synthetic oracle features, not model results."""
    from dataclasses import replace
    frames = np.zeros((64, SIZE, SIZE, 3), np.uint8)
    initial_mask = np.zeros((SIZE, SIZE), np.uint8)
    initial_mask[64:96, 64:96] = 1
    initial_mask[160:192, 64:96] = 2
    labels = initial_patch_labels(initial_mask)
    features = np.zeros((32, GRID, GRID, 5), np.float32)
    features[..., 0] = 1.
    for step in range(32):
        col = 4+step % 10
        features[step, 4:6, col:col+2] = (0, 1, 0, 0, 0)
        features[step, 10:12, col:col+2] = (0, 0, 1, 0, 0)
    config = replace(TrackerConfig(), context_weight=0., motion_weight=0.)
    pred = track_parts(features, frames, initial_mask, config, labels)
    assert pred.cells.shape == (32, 2, 2)
    for step in range(32):
        for ki, row in enumerate((4, 10)):
            assert row <= pred.cells[step, ki, 0] < row+2
            assert 4+step % 10 <= pred.cells[step, ki, 1] < 6+step % 10
    altered = features.copy()
    altered[16:] = np.random.default_rng(12).normal(size=altered[16:].shape)
    changed = track_parts(altered, frames, initial_mask, config, labels)
    assert np.array_equal(pred.cells[:16], changed.cells[:16])
    assert np.array_equal(pred.scores[:16], changed.scores[:16])
    assert np.isfinite(pred.scores).all()
    # Model-independent baseline should find known stationary coloured parts.
    image = np.full((SIZE, SIZE, 3), 220, np.uint8)
    image[64:96, 64:96] = (40, 100, 180)
    image[160:192, 64:96] = (180, 80, 40)
    classic = template_tracker(np.repeat(image[None], 4, axis=0), initial_mask,
                               TemplateConfig(scales=(1.,)))
    for ki, row in enumerate((4, 10)):
        assert np.all((classic.cells[:, ki, 0] >= row) & (classic.cells[:, ki, 0] < row+2))
        assert np.all((classic.cells[:, ki, 1] >= 4) & (classic.cells[:, ki, 1] < 6))
    return dict(dynamic_time_shape=True, oracle_translation=True,
                tracker_future_feature_independence=True, finite_output=True,
                stationary_rgb_template=True,
                warning="Oracle implementation checks are not pretrained-model results.")
