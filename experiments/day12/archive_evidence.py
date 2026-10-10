"""Pack or recover Day12 evidence, checking every part and payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def child(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root):
        raise ValueError('Evidence path escapes its root')
    return p


def pack(source, out, kind):
    if Path(out).exists() or Path(out).is_symlink():
        raise ValueError('Archive output must be new')
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_dir() or out.exists() or out.is_relative_to(source):
        raise ValueError('Require an input directory and a new output outside it')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', kind or ''):
        raise ValueError('Archive kind must be a nonempty filename-safe label')
    blocked = {'models', 'model-cache', 'offload', 'prompt-staging', '.cache', '.git'}
    prompt_caches = {'prompts/prompt_14000.pt', 'prompts/prompt_14001.pt'}
    files = []
    for path in sorted(source.rglob('*')):
        relative = str(path.relative_to(source))
        if path.is_symlink() or child(source, relative) != path:
            raise ValueError('Input links are not permitted: ' + relative)
        if blocked.intersection(path.relative_to(source).parts):
            raise ValueError('Model/offload directories are not evidence: ' + relative)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('Input is not a regular file: ' + relative)
        if (path.suffix.lower() in {'.safetensors', '.bin', '.ckpt', '.pth', '.pt2', '.onnx'}
                or (path.suffix.lower() == '.pt' and
                    (relative not in prompt_caches or path.stat().st_size > 3 * 1024**2))):
            raise ValueError('Model weights are not evidence: ' + relative)
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path)))
    if not files:
        raise ValueError('No evidence files to archive')
    # Build and verify before creating the final output directory. Normalized
    # metadata makes archive bytes independent of local ownership and timestamps.
    with tempfile.NamedTemporaryFile(suffix='.tar.xz') as archive:
        with tarfile.open(archive.name, 'w:xz', preset=3, dereference=True) as tf:
            for row in files:
                path = child(source, row['path'])
                info = tf.gettarinfo(str(path), arcname=row['path'])
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode, info.pax_headers = 0o644, {}
                with path.open('rb') as handle:
                    tf.addfile(info, handle)
        with tarfile.open(archive.name, 'r:xz') as tf:
            if tf.getnames() != [row['path'] for row in files]:
                raise ValueError('Packed entry inventory changed')
            for row in files:
                item, digest = tf.getmember(row['path']), hashlib.sha256()
                if not item.isfile() or item.size != row['bytes']:
                    raise ValueError('Packed file type/size changed')
                with tf.extractfile(item) as handle:
                    for block in iter(lambda: handle.read(1 << 20), b''):
                        digest.update(block)
                if digest.hexdigest() != row['sha256']:
                    raise ValueError('Input changed while packing: ' + row['path'])
        manifest = dict(version='day12_recovered_evidence_archive_v1', kind=kind,
                        complete=True, archive_bytes=Path(archive.name).stat().st_size,
                        archive_sha256=sha256(archive.name), files=files, parts=[])
        out.mkdir(parents=True)  # No exist_ok: never replace an existing archive.
        with Path(archive.name).open('rb') as handle:
            for index, data in enumerate(iter(lambda: handle.read(6 * 1024**2), b'')):
                name = f'{kind}.tar.xz.part{index:04d}'
                with (out / name).open('xb') as part:
                    part.write(data)
                manifest['parts'].append(dict(path=name, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    git_blob_sha1=hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()))
        with (out / 'archive_manifest.json').open('x') as handle:
            handle.write(json.dumps(manifest, indent=2) + '\n')
    return dict(complete=True, files=len(files), parts=len(manifest['parts']),
                archive_bytes=manifest['archive_bytes'], archive_sha256=manifest['archive_sha256'], kind=kind)


def unpack(source, out):
    source, out = Path(source), Path(out)
    m = json.loads((source / 'archive_manifest.json').read_text())
    if not m['complete'] or m['version'] != 'day12_recovered_evidence_archive_v1':
        raise ValueError('Unexpected archive')
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.tar.xz') as f:
        for part in m['parts']:
            p = child(source, part['path'])
            if sha256(p) != part['sha256'] or p.stat().st_size != part['bytes']:
                raise ValueError('Archive part mismatch')
            f.write(p.read_bytes())
        f.flush()
        if sha256(f.name) != m['archive_sha256'] or Path(f.name).stat().st_size != m['archive_bytes']:
            raise ValueError('Reassembled archive mismatch')
        with tarfile.open(f.name) as tf:
            if set(tf.getnames()) != {r['path'] for r in m['files']}:
                raise ValueError('Archive entry set mismatch')
            for item in tf.getmembers():
                child(out, item.name)
                if not item.isfile():
                    raise ValueError('Unexpected archive member type')
            tf.extractall(out, filter='data')
    for row in m['files']:
        p = child(out, row['path'])
        if sha256(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise ValueError('Extracted payload mismatch')
    return {'complete': True, 'files': len(m['files']), 'kind': m['kind']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['pack', 'unpack'])
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--kind', help='Required for pack; e.g. actual_kaggle_video_results')
    a = p.parse_args()
    if a.mode == 'pack' and not a.kind:
        p.error('--kind is required for pack')
    result = pack(a.source, a.out, a.kind) if a.mode == 'pack' else unpack(a.source, a.out)
    print(json.dumps(result))
