"""Plot saved predictions; does not run trackers or alter frozen configuration."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frames-root', type=Path, required=True, help='Contains car-roundabout/frames and car-shadow/frames')
parser.add_argument('--roundabout-results', type=Path, default=HERE / 'car_roundabout')
parser.add_argument('--shadow-results', type=Path, default=HERE / 'car_shadow')
parser.add_argument('--output', type=Path, default=HERE / 'two_clip_part_tracking_comparison.png')
args = parser.parse_args()
sys.path.insert(0, str(HERE.parent))
from real_video import polygon_mask

CLIPS = [
    ('Development: car-roundabout', 'car-roundabout',
     args.roundabout_results, 'car_roundabout_annotations.json', 63),
    ('Frozen configuration: car-shadow', 'car-shadow',
     args.shadow_results, 'car_shadow_annotations.json', 39),
]
METHODS = [('SAM2.1 tiny', 'sam2_masks.npz'), ('Per-part LK affine', 'lk_masks.npz'),
           ('Shared-plane homography', 'shared_homography_masks.npz')]
COLORS = ['#ff9d24', '#13d8ee']


def truth_at(annotations, t, ids, shape):
    return np.asarray([polygon_mask(annotations['frames'][str(t)][str(i)], shape) for i in ids])


def contour(ax, mask, color, width=1.3):
    if mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=width)


def crop(ax, masks):
    yy, xx = np.where(masks.any(axis=0))
    cx, cy = (xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2
    # Same pixel dimensions on every panel, retaining visible scale change.
    ax.set_xlim(cx - 135, cx + 135)
    ax.set_ylim(cy + 105, cy - 105)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(2, 4, figsize=(17.6, 8.8), dpi=160)
fig.patch.set_facecolor('#f5f7fa')
for r, (label, sequence, folder, annotation_name, t) in enumerate(CLIPS):
    annotations = json.loads((HERE.parent / annotation_name).read_text())
    rawdir = args.frames_root / sequence / 'frames'
    raw0 = cv2.cvtColor(cv2.imread(str(rawdir / '00000.jpg')), cv2.COLOR_BGR2RGB)
    raw = cv2.cvtColor(cv2.imread(str(rawdir / f'{t:05d}.jpg')), cv2.COLOR_BGR2RGB)
    ids = sorted(p['id'] for p in annotations['parts'])
    truth0 = truth_at(annotations, 0, ids, raw.shape[:2])
    truth = truth_at(annotations, t, ids, raw.shape[:2])
    ax = axes[r, 0]
    ax.imshow(raw0)
    for j in range(len(ids)):
        contour(ax, truth0[j], COLORS[j], 1.6)
    crop(ax, truth0)
    ax.set_title(f'{label}\nFrame 0: supplied part prompts', fontsize=11, loc='left', pad=9)
    for c, (method, filename) in enumerate(METHODS, 1):
        saved = np.load(folder / filename)
        prediction = saved['masks'][t]
        order = saved['ids'].tolist()
        prediction = prediction[[order.index(i) for i in ids]]
        ax = axes[r, c]
        ax.imshow(raw)
        scores = []
        for j in range(len(ids)):
            contour(ax, prediction[j], COLORS[j])
            contour(ax, truth[j], '#62f66a', .85)
            scores.append((truth[j] & prediction[j]).sum() / (truth[j] | prediction[j]).sum())
        crop(ax, truth)
        ax.set_title(f'{method} | frame {t}\nDoor IoU {scores[0]:.1%}  ·  Window IoU {scores[1]:.1%}',
                     fontsize=11, loc='left', pad=9)
for ax in axes.ravel():
    for spine in ax.spines.values():
        spine.set_edgecolor('#d8dfe8')
fig.suptitle('Part tracking: inspect the final evaluated frame of each clip', x=.024, y=.99,
             ha='left', fontsize=17, fontweight='bold', color='#172334')
fig.text(.024, .945, 'Green: approximate manual reference contour   |   Orange: door prediction   |   Cyan: window prediction',
         fontsize=11, color='#344255')
fig.text(.024, .035, 'All methods receive frame-0 masks only. Same pixel scale in every panel. '
         'Sparse polygons are approximate, not pixel-perfect ground truth. No scored disappearance/reappearance events.',
         fontsize=10, color='#344255')
fig.subplots_adjust(left=.024, right=.995, top=.86, bottom=.075, wspace=.05, hspace=.28)
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
plt.close(fig)
print(output)
