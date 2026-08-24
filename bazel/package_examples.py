#!/usr/bin/env python3
"""Build a deterministic examples archive from tracked sources and model weights."""

import argparse
import gzip
import io
import os
from pathlib import Path
import subprocess
import tarfile


def tracked_example_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "examples"],
    )
    return [Path(path.decode()) for path in output.split(b"\0") if path]


def tar_info(name: str, mode: int, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode = mode
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_archive(archive: Path, weights: Path) -> dict[str, Path]:
    files = tracked_example_files()
    model_path = Path("examples/nanogpt/out-shakespeare-char/model.safetensors")
    if model_path in files:
        raise RuntimeError(f"{model_path} must not be tracked by Git")
    files.append(model_path)
    sources = {path.as_posix(): weights if path == model_path else path for path in files}

    directories = {Path("examples")}
    for path in files:
        directories.update(path.parents)
    directories.discard(Path("."))

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for directory in sorted(directories, key=lambda path: path.as_posix()):
                    info = tar_info(directory.as_posix() + "/", 0o755)
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)

                for path in sorted(files, key=lambda item: item.as_posix()):
                    source = weights if path == model_path else path
                    data = source.read_bytes()
                    mode = 0o755 if os.access(source, os.X_OK) else 0o644
                    tar.addfile(tar_info(path.as_posix(), mode, len(data)), io.BytesIO(data))
    return sources


def verify_archive(archive: Path, sources: dict[str, Path]) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        if set(members) != set(sources):
            missing = sorted(set(sources) - set(members))
            unexpected = sorted(set(members) - set(sources))
            raise RuntimeError(
                f"archive manifest mismatch: missing={missing}, unexpected={unexpected}",
            )
        for name, source in sources.items():
            archived = tar.extractfile(members[name])
            if archived is None or archived.read() != source.read_bytes():
                raise RuntimeError(f"archive content mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    sources = build_archive(args.archive, args.weights)
    verify_archive(args.archive, sources)


if __name__ == "__main__":
    main()
