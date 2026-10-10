"""Cache real frozen Wan VAE targets for the hash-verified native-image dataset."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from huggingface_hub import hf_hub_download

from data import NativeImageDataset, TARGET_VERSION, sha256
from vace_bridge import MODEL_ID, normalize_wan_latents

REVISION = 'ec4d2cb062b548996b179d493fdd05340de702a1'


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [NativeImageDataset(args.cache_root, s) for s in ('train', 'dev')]
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder='vae', revision=REVISION,
        cache_dir=args.model_cache, torch_dtype=torch.float32).eval().requires_grad_(False).to(args.device)
    files = {}
    for name in ('config.json', 'diffusion_pytorch_model.safetensors'):
        p = hf_hub_download(MODEL_ID, 'vae/' + name, revision=REVISION, cache_dir=args.model_cache)
        files[name] = {'sha256': sha256(p), 'bytes': Path(p).stat().st_size}
    provenance = {
        'model_id': MODEL_ID, 'revision': REVISION, 'files': files,
        'class': 'AutoencoderKLWan', 'precision': 'float32', 'posterior': 'mode',
        'normalization': '(posterior_mode - latents_mean) / latents_std',
        'latents_mean': list(vae.config.latents_mean), 'latents_std': list(vae.config.latents_std),
        'input': '[1,3,1,384,384] RGB in [-1,1]', 'output': '[16,48,48]',
    }
    manifest_path = out / 'latent_manifest.json'
    base = {'version': TARGET_VERSION, 'jepa_manifest_sha256': datasets[0].manifest_sha256,
            'vae': provenance, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
            'device': args.device, 'threads': args.threads}, 'items': []}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else base
    for key in ('version', 'jepa_manifest_sha256', 'vae', 'runtime'):
        if manifest[key] != base[key]:
            raise ValueError('Cannot mix target caches: ' + key)
    write_json(manifest_path, manifest)
    with torch.inference_mode():
        empty = normalize_wan_latents(vae, vae.encode(torch.zeros(1, 3, 1, 384, 384, device=args.device)).latent_dist.mode())
        np.save(out / 'normalized_empty.npy', empty.cpu().numpy())
        manifest['empty_video_latent'] = {'path': 'normalized_empty.npy', 'sha256': sha256(out / 'normalized_empty.npy')}
        for dataset in datasets:
            for index, record in enumerate(dataset.records):
                old = next((x for x in manifest['items'] if x['sample_id'] == record.sample_id), None)
                if old:
                    if sha256(out / old['path']) != old['sha256']:
                        raise ValueError('Cached target changed')
                    continue
                started = time.perf_counter()
                example = dataset[index]
                raw = vae.encode(example['rgb'][None, :, None].to(args.device)).latent_dist.mode()
                latent = normalize_wan_latents(vae, raw).cpu().numpy()[0, :, 0]
                if latent.shape != (16, 48, 48) or not np.isfinite(latent).all():
                    raise ValueError('Unexpected latent')
                name = record.sample_id + '.npy'
                np.save(out / name, latent)
                manifest['items'].append({'sample_id': record.sample_id, 'path': name,
                    'sha256': sha256(out / name), 'rgb_sha256': record.rgb_sha256,
                    'seconds': time.perf_counter() - started})
                write_json(manifest_path, manifest)
                print('TARGET', record.sample_id, round(time.perf_counter() - started, 3), flush=True)
    manifest['complete'] = len(manifest['items']) == 80
    write_json(manifest_path, manifest)
    print('COMPLETE', manifest['complete'], sha256(manifest_path), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-cache', default='/tmp/day11-model')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())
