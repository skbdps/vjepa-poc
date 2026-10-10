"""Recover the preserved Day12 outputs, checking every part and payload."""
import argparse
import hashlib
import json
from pathlib import Path
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
    p.add_argument('mode', choices=['unpack'])
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(unpack(a.source, a.out)))
