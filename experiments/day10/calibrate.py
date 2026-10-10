"""Native-image interface/readout check on a separate calibration seed."""
from pathlib import Path
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import torch

DAY8 = Path(__file__).resolve().parents[1] / 'day8'
sys.path.insert(0, str(DAY8))
import data
import extract
import probe
import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    encoder = extract.load_encoder(a.upstream, 'cpu')
    readout, _ = probe.load_probe(a.probe)
    pair = data.generate_pair({'seed': 13000, 'dx': 48, 'name': 'image_calibration', 'split': 'calibration'})
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    rows, arrays = [], {}
    for name, image_key, mask_key in [('source', 'frames_source', 'masks_source'), ('target', 'frames_target', 'masks_target')]:
        image = pair[image_key][15]
        mask = pair[mask_key][15]
        fraction = mask.reshape(24, 16, 24, 16).mean(axis=(1, 3)).astype(np.float32)[None]
        tick = time.perf_counter()
        x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None, :, None].float() / 255
        with torch.inference_mode():
            z = encoder((x-mean)/std)
        assert z.shape == (1, 576, 1024) and torch.isfinite(z).all()
        tokens = z.numpy().reshape(1, 24, 24, 1024)
        pred = probe.predict_probe(readout, tokens)
        lane = evaluate._lane_mask(fraction)
        actual = evaluate.centroid(fraction[0])
        predicted = evaluate.centroid(pred['occupancy'][0], lane, .25)
        rows.append({'name': name, 'seconds': time.perf_counter()-tick,
                     'input_shape': list(x.shape), 'output_shape': list(z.shape),
                     'selected_centroid_error_px': evaluate._centroid_error(predicted, actual),
                     'selected_iou': evaluate._iou(pred['occupancy'][0], fraction[0], lane)[0],
                     'centroid_present': predicted[0] is not None,
                     'source_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        arrays[name+'_tokens'] = tokens
        arrays[name+'_rgb'] = pred['rgb']
        arrays[name+'_occupancy'] = pred['occupancy']
    cache = a.out/'native_image_calibration.npz'
    np.savez_compressed(cache, **arrays)
    record = {'seed': 13000, 'frame_index': 15, 'dx': 48,
              'precision': 'float32', 'device': 'cpu', 'torch': torch.__version__,
              'numpy': np.__version__, 'test_images_accessed': False,
              'probe_sha256': extract.sha256(a.probe),
              'cache_sha256': extract.sha256(cache),
              'rows': rows,
              'native_image_interface_pass': all(r['output_shape'] == [1,576,1024] for r in rows),
              'selected_localization_pass': all(r['centroid_present'] and r['selected_centroid_error_px'] < 16 for r in rows),
              'scope': 'Engineering interface and frozen-readout localization check only. IoU is descriptive; this is not a held-out edit result.'}
    extract.write_json(a.out/'native_image_calibration.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
