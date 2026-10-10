"""Hash-bound, genuine native-image JEPA examples for the Day11 bridge.

This module reads only the 40 entries in the Day10 training/development
manifest. It never globs cache files, opens previous test scenes, or feeds
renderer metadata/masks to the translator. Full-resolution RGB is regenerated
only as separately identified VAE supervision and checked against the original
raw-pixel hashes. No JEPA encoder inference is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[2]
CACHE_VERSION = 'day10_native_image_readout_followup_v1'
TARGET_VERSION = 'day11_wan_latent_targets_v1'
GRID, CHANNELS, SIZE, FRAME_INDEX = 24, 1024, 384, 15
SPLIT_SEEDS = {'train': tuple(range(13400, 13432)),
               'dev': tuple(range(13500, 13508))}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _child_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError('Manifest paths must be relative')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes its root')
    return resolved


def _legacy_renderer():
    # Day11 is also named data.py, so a normal import would select this module.
    path = REPO / 'experiments/day8/data.py'
    spec = importlib.util.spec_from_file_location('day11_pinned_day8_renderer', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load the recorded procedural renderer')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    scene: str
    seed: int
    split: str
    view_index: int
    view: str
    rgb_sha256: str
    cache_path: str
    cache_sha256: str


class NativeImageDataset(Dataset):
    """64 train or 16 development images from the existing genuine caches.

    ``jepa`` is the condition: float32 [1024,24,24]. ``rgb`` is supervision:
    float32 [3,384,384] in [-1,1]. Never pass ``rgb`` to the JEPA projector.
    Scene-level caching avoids repeatedly rendering all 32 legacy frames.
    """

    def __init__(self, cache_root: str | Path, split: str,
                 verify_hashes: bool = True, cache_scenes: bool = True):
        if split not in SPLIT_SEEDS:
            raise ValueError('Only the previously declared train/dev splits are available')
        if not verify_hashes:
            raise ValueError('Hash verification is mandatory for the bridge dataset')
        self.root = Path(cache_root).resolve()
        self.split = split
        self.cache_scenes = cache_scenes
        self.manifest_path = self.root / 'features_manifest.json'
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('version') != CACHE_VERSION:
            raise ValueError('Not the genuine native-image training manifest')
        if (self.manifest.get('test_accessed') is not False
                or self.manifest.get('precision') != 'float32'
                or self.manifest.get('cache_precision') != 'float32'
                or self.manifest.get('modality') != 'native single image [1,3,1,384,384]'):
            raise ValueError('Unexpected cache provenance or modality')
        renderer_relative = 'experiments/day8/data.py'
        renderer_digest = self.manifest.get('source_hashes', {}).get(renderer_relative)
        if renderer_digest != sha256(REPO / renderer_relative):
            raise ValueError('The procedural renderer differs from the cached run')
        all_rows = self.manifest.get('scenes', [])
        if len(all_rows) != 40:
            raise ValueError('The native training/development manifest must contain 40 scenes')
        expected = self.manifest.get('train_specs', []) + self.manifest.get('dev_specs', [])
        if [row['spec'] for row in all_rows] != expected:
            raise ValueError('Manifest scenes differ from the predeclared specifications')
        for name, seeds in SPLIT_SEEDS.items():
            actual = [row['spec']['seed'] for row in all_rows
                      if row['spec']['split'] == name]
            if actual != list(seeds):
                raise ValueError('Unexpected or reordered scene seeds: ' + name)
        self.rows = [row for row in all_rows if row['spec']['split'] == split]
        self.records: list[ImageRecord] = []
        self._row_by_scene = {}
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._renderer = _legacy_renderer()
        for row in self.rows:
            spec = row['spec']
            if spec.get('frame_index') != FRAME_INDEX or spec.get('views') != ['source', 'genuine_shifted_target']:
                raise ValueError('Unexpected image view or frame selection')
            if spec['name'] in self._row_by_scene:
                raise ValueError('Duplicate scene')
            self._row_by_scene[spec['name']] = row
            path = _child_path(self.root, row['path'])
            if sha256(path) != row['sha256']:
                raise ValueError('JEPA feature file changed: ' + str(path))
            for index, view in enumerate(('source', 'target')):
                self.records.append(ImageRecord(
                    sample_id=spec['name'] + '__' + view,
                    scene=spec['name'], seed=int(spec['seed']), split=split,
                    view_index=index, view=view,
                    rgb_sha256=row[view + '_rgb_sha256'],
                    cache_path=row['path'], cache_sha256=row['sha256']))

    def __len__(self) -> int:
        return len(self.records)

    def _scene(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        row = self._row_by_scene[name]
        path = _child_path(self.root, row['path'])
        # Recheck at time of first use rather than relying on construction alone.
        if sha256(path) != row['sha256']:
            raise ValueError('JEPA feature file changed after dataset construction')
        with np.load(path, allow_pickle=False) as archive:
            tokens = archive['tokens'].copy()
        if tokens.shape != (2, 1, GRID, GRID, CHANNELS) or tokens.dtype != np.float32:
            raise ValueError('Unexpected genuine JEPA tensor shape or precision')
        if not np.isfinite(tokens).all():
            raise ValueError('Nonfinite JEPA features')
        pair = self._renderer.generate_pair(row['spec'])
        images = np.stack([pair['frames_source'][FRAME_INDEX],
                           pair['frames_target'][FRAME_INDEX]])
        del pair
        for index, view in enumerate(('source', 'target')):
            if array_sha256(images[index]) != row[view + '_rgb_sha256']:
                raise ValueError('Regenerated RGB differs from original cache: ' + name + '/' + view)
        # Discard every scoring mask, metadata field, and extra frame.
        result = (np.ascontiguousarray(tokens[:, 0].transpose(0, 3, 1, 2)), images)
        if self.cache_scenes:
            self._cache[name] = result
        return result

    def rgb_uint8(self, index: int) -> np.ndarray:
        record = self.records[index]
        return self._scene(record.scene)[1][record.view_index].copy()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        features, images = self._scene(record.scene)
        return {
            'sample_id': record.sample_id, 'scene': record.scene,
            'seed': record.seed, 'view_index': record.view_index,
            'rgb_sha256': record.rgb_sha256,
            'jepa': torch.from_numpy(features[record.view_index].copy()),
            'rgb': torch.from_numpy(images[record.view_index].copy())
                .permute(2, 0, 1).float().div(127.5).sub(1),
        }

    def provenance(self) -> dict[str, Any]:
        return {
            'split': self.split, 'scene_count': len(self.rows), 'image_count': len(self),
            'manifest_sha256': self.manifest_sha256,
            'renderer_sha256': sha256(REPO / 'experiments/day8/data.py'),
            'sample_ids': [record.sample_id for record in self.records],
            'condition': 'full dense genuine JEPA features only',
            'supervision': 'separately regenerated and raw-pixel-hash-verified RGB',
            'forbidden_condition_inputs': ['RGB target', 'target mask', 'seed',
                                           'renderer metadata', 'requested shift'],
        }


def train_channel_statistics(dataset: NativeImageDataset,
                             std_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Streaming, channelwise statistics from training features only."""
    if dataset.split != 'train':
        raise ValueError('Development/test data cannot set normalization')
    if std_floor <= 0:
        raise ValueError('Standard-deviation floor must be positive')
    total = np.zeros(CHANNELS, np.float64)
    squared = np.zeros(CHANNELS, np.float64)
    count = 0
    for record in dataset.records:
        tokens = dataset._scene(record.scene)[0][record.view_index]
        flattened = tokens.reshape(CHANNELS, -1).astype(np.float64)
        total += flattened.sum(axis=1)
        squared += np.square(flattened).sum(axis=1)
        count += flattened.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 0))
    return mean.astype(np.float32), np.maximum(std, std_floor).astype(np.float32)


class LatentTargetStore:
    """Load real frozen-VAE targets independently from JEPA conditions.

    Expected manifest: version, jepa_manifest_sha256, vae (provenance object),
    items [{sample_id,path,sha256,rgb_sha256}]. Each .npy or single-key .npz
    stores float32 normalized Wan latents [16,48,48] or [16,1,48,48].
    It is deliberately not a method on the condition dataset.
    """

    def __init__(self, root: str | Path, jepa_manifest_sha256: str,
                 manifest_name: str = 'latent_manifest.json'):
        self.root = Path(root).resolve()
        self.manifest_path = _child_path(self.root, manifest_name)
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('version') != TARGET_VERSION:
            raise ValueError('Unexpected VAE target manifest version')
        if self.manifest.get('jepa_manifest_sha256') != jepa_manifest_sha256:
            raise ValueError('VAE targets are not bound to these JEPA/RGB pairs')
        if not isinstance(self.manifest.get('vae'), dict) or not self.manifest['vae']:
            raise ValueError('Missing frozen VAE provenance')
        items = self.manifest.get('items', [])
        self.items = {item['sample_id']: item for item in items}
        if len(self.items) != len(items):
            raise ValueError('Duplicate VAE target sample identifier')
        self._cache = {}

    def get(self, sample_id: str, rgb_sha256: str) -> torch.Tensor:
        item = self.items[sample_id]
        if item['rgb_sha256'] != rgb_sha256:
            raise ValueError('VAE target and RGB supervision do not match')
        if sample_id not in self._cache:
            path = _child_path(self.root, item['path'])
            if sha256(path) != item['sha256']:
                raise ValueError('VAE target file changed')
            if path.suffix == '.npy':
                latent = np.load(path, allow_pickle=False)
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    if archive.files != ['latent']:
                        raise ValueError('Target archive must contain only latent')
                    latent = archive['latent'].copy()
            else:
                raise ValueError('Only non-pickle numpy target files are accepted')
            if latent.shape == (16, 1, 48, 48):
                latent = latent[:, 0]
            if latent.shape != (16, 48, 48) or latent.dtype != np.float32:
                raise ValueError('Unexpected normalized Wan target shape/precision')
            if not np.isfinite(latent).all():
                raise ValueError('Nonfinite VAE target')
            self._cache[sample_id] = np.ascontiguousarray(latent)
        return torch.from_numpy(self._cache[sample_id].copy())

    def validate_dataset_coverage(self, dataset: NativeImageDataset) -> None:
        for record in dataset.records:
            self.get(record.sample_id, record.rgb_sha256)
