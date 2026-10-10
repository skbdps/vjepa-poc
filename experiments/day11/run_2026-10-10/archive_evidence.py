#!/usr/bin/env python3
"""Pack, verify, and split a completed evidence directory without changing it.

Example:
  python /tmp/archive_evidence.py --input /path/to/evidence \
    --out /tmp/evidence_archive --name evidence --part-size-mib 4

Only regular files and directories are accepted. Symlinks and special files
are rejected. Output must be an empty/new directory outside the input tree.
All archive member and manifest artifact paths are relative. Source content
is SHA256-hashed before packing, every archived file is hashed by reading the
compressed archive, and every written part is reread and checked. Concatenated
part hashes are checked against the full archive hash. No extraction occurs.
The source must remain unchanged during packing; this is not a live snapshot.

Default raw parts are 4MiB, allowing headroom for encoding/request overhead.
The hard upper bound is 8MiB per raw part. Git blob SHA1 uses the exact bytes:
SHA1(b'blob '+str(size).encode()+b'\\0'+content), suitable for GitHub validation.
The scientific code and experiment outputs are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

BUFFER = 1024 * 1024
MAX_PART_BYTES = 8 * 1024 * 1024
VERSION = "verified_evidence_tarxz_parts_v1"


def fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def metadata(value):
    return {"mode": stat.S_IMODE(value.st_mode), "uid": value.st_uid,
            "gid": value.st_gid, "mtime_ns": value.st_mtime_ns}


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Unsafe/noncanonical relative path: " + repr(name))
    return path.as_posix()


def inventory(root):
    records = {}
    def visit(directory):
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda item: item.name)
        for entry in entries:
            path = Path(entry.path)
            name = safe_relative(path.relative_to(root).as_posix())
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Symlink rejected: " + name)
            if stat.S_ISDIR(info.st_mode):
                kind = "directory"
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
            else:
                raise ValueError("Non-regular filesystem entry rejected: " + name)
            records[name] = {"path": name, "kind": kind, **metadata(info),
                             "bytes": info.st_size if kind == "file" else 0,
                             "_fingerprint": fingerprint(info)}
            if kind == "directory":
                visit(path)
    visit(root)
    return records


@contextmanager
def unchanged_file(root, record):
    path = root / record["path"]
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            if fingerprint(os.fstat(handle.fileno())) != record["_fingerprint"]:
                raise ValueError("Source changed before read: " + record["path"])
            yield handle
            if fingerprint(os.fstat(handle.fileno())) != record["_fingerprint"]:
                raise ValueError("Source changed during read: " + record["path"])
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stream_hashes(handle, size, joined=None):
    digest, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(b"blob " + str(size).encode("ascii") + b"\0")
    count = 0
    for block in iter(lambda: handle.read(BUFFER), b""):
        count += len(block)
        digest.update(block)
        blob.update(block)
        if joined is not None:
            joined.update(block)
    if count != size:
        raise ValueError(f"Read byte count differs: {count} != {size}")
    return {"bytes": count, "sha256": digest.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def file_hashes(path, joined=None):
    with Path(path).open("rb") as handle:
        return stream_hashes(handle, os.fstat(handle.fileno()).st_size, joined)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def pack(source, out, name="evidence", part_bytes=4 * 1024 * 1024, preset=6):
    source, out = Path(source), Path(out)
    if source.is_symlink() or out.is_symlink():
        raise ValueError("Input/output root symlinks are not accepted")
    source, out = source.resolve(), out.resolve()
    if not source.is_dir():
        raise ValueError("Input must be a directory")
    if out == source or out.is_relative_to(source):
        raise ValueError("Output must be outside the input evidence tree")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Name must contain only letters, digits, dot, dash or underscore")
    if not isinstance(part_bytes, int) or not 1 <= part_bytes <= MAX_PART_BYTES:
        raise ValueError("Raw part bytes must be between1 and8MiB")
    if not 0 <= preset <= 9:
        raise ValueError("XZ preset must be in0..9")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be a new or empty directory; nothing is overwritten")
    out.mkdir(parents=True, exist_ok=True)
    records = inventory(source)
    root_metadata = metadata(source.stat())
    files = [value for value in records.values() if value["kind"] == "file"]
    for record in files:
        with unchanged_file(source, record) as handle:
            record.update(stream_hashes(handle, record["bytes"]))
    archive_name = name + ".tar.xz"
    archive = out / archive_name
    temporary = out / (archive_name + ".partial")
    print("HASHED", len(files), "files", sum(r["bytes"] for r in files), "bytes", flush=True)
    with tarfile.open(temporary, "x:xz", format=tarfile.PAX_FORMAT,
                      dereference=True, preset=preset) as tar:
        for path, record in records.items():
            if fingerprint((source / path).lstat()) != record["_fingerprint"]:
                raise ValueError("Source changed before packing: " + path)
            info = tar.gettarinfo(str(source / path), arcname=path)
            info.pax_headers["mtime"] = str(Decimal(record["mtime_ns"]) / Decimal(10**9))
            if record["kind"] == "directory":
                tar.addfile(info)
            else:
                if not info.isfile() or info.size != record["bytes"]:
                    raise ValueError("Packing entry changed type/size: " + path)
                with unchanged_file(source, record) as handle:
                    tar.addfile(info, handle)
    after = inventory(source)
    if set(after) != set(records) or any(after[p]["_fingerprint"] != r["_fingerprint"] for p, r in records.items()):
        raise ValueError("Input tree changed while it was being archived")
    seen = set()
    with tarfile.open(temporary, "r:xz") as tar:
        for member in tar:
            path = safe_relative(member.name.rstrip("/"))
            if path in seen or path not in records:
                raise ValueError("Duplicate/unexpected archive member: " + path)
            seen.add(path)
            expected = records[path]
            if member.isdir() and expected["kind"] == "directory":
                continue
            if not member.isfile() or expected["kind"] != "file":
                raise ValueError("Archive contains unsupported/mismatched member: " + path)
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError("Archive file cannot be read: " + path)
            with handle:
                actual = stream_hashes(handle, member.size)
            if any(actual[key] != expected[key] for key in ("bytes", "sha256", "git_blob_sha1")):
                raise ValueError("Archived bytes differ from input: " + path)
    if seen != set(records):
        raise ValueError("Archive omitted input entries")
    temporary.replace(archive)
    print("VERIFIED_ARCHIVE_CONTENTS", len(files), "files", flush=True)
    total_size = archive.stat().st_size
    whole, whole_blob = hashlib.sha256(), hashlib.sha1()
    whole_blob.update(b"blob " + str(total_size).encode("ascii") + b"\0")
    parts, offset = [], 0
    with archive.open("rb") as handle:
        while True:
            block = handle.read(part_bytes)
            if not block:
                break
            part_name = archive_name + f".part{len(parts):04d}"
            path = out / part_name
            with path.open("xb") as part:
                part.write(block)
            whole.update(block)
            whole_blob.update(block)
            actual = file_hashes(path)
            if actual["bytes"] != len(block) or actual["sha256"] != hashlib.sha256(block).hexdigest():
                raise ValueError("Written part differs from archive bytes: " + part_name)
            parts.append({"path": part_name, "index": len(parts), "offset": offset, **actual})
            offset += len(block)
    if offset != total_size:
        raise ValueError("Archive byte count changed while splitting")
    joined = hashlib.sha256()
    for part in parts:
        if file_hashes(out / part["path"], joined) != {k: part[k] for k in ("bytes", "sha256", "git_blob_sha1")}:
            raise ValueError("Part verification failed: " + part["path"])
    if joined.hexdigest() != whole.hexdigest():
        raise ValueError("Concatenated parts do not reconstruct the archive")
    public_records = [{k: v for k, v in record.items() if not k.startswith("_")} for record in records.values()]
    manifest = {"version": VERSION, "complete": True, "source_directory_name": source.name,
                "source_root_metadata": root_metadata, "format": "POSIX pax tar compressed losslessly with XZ",
                "compression_preset": preset, "helper_sha256": file_hashes(Path(__file__))["sha256"],
                "archive": {"path": archive_name, "bytes": total_size, "sha256": whole.hexdigest(),
                            "git_blob_sha1": whole_blob.hexdigest()},
                "part_bytes_limit": part_bytes, "hard_part_bytes_limit": MAX_PART_BYTES,
                "parts": parts, "part_count": len(parts), "file_count": len(files),
                "directory_count": len(records)-len(files), "uncompressed_file_bytes": sum(r["bytes"] for r in files),
                "entries": public_records,
                "verification": {"input_symlinks": "rejected", "input_special_files": "rejected",
                                 "source_unchanged_during_pack": True, "every_archived_file_read_and_hash_verified": True,
                                 "every_part_reread_and_hash_verified": True, "concatenated_parts_sha256_matches_archive": True},
                "reassembly": "Concatenate parts in ascending index order; verify archive SHA256; read with tarfile/tar -xJf.",
                "git_blob_hash_definition": "SHA1('blob '+decimal_byte_count+'\\0'+exact_file_bytes)"}
    manifest_path = out / (name + ".manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": file_hashes(manifest_path)["sha256"],
                      "archive_bytes": total_size, "parts": len(parts), "complete": True}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Completed evidence directory")
    parser.add_argument("--out", required=True, help="New/empty directory outside input")
    parser.add_argument("--name", default="evidence")
    parser.add_argument("--part-size-mib", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--preset", type=int, default=6, choices=range(10))
    args = parser.parse_args()
    try:
        pack(args.input, args.out, args.name, args.part_size_mib*1024*1024, args.preset)
    except Exception as error:
        print(json.dumps({"complete": False, "error": type(error).__name__, "message": str(error)}), flush=True)
        raise


if __name__ == "__main__":
    main()
