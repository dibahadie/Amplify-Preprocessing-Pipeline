#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from pysndfile import sndio


def audio_hash(path: Path) -> str:
    data, *_ = sndio.read(str(path))
    return hashlib.md5(data.tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply manual transcript overrides only when the current split audio "
            "matches a recorded decoded-audio hash."
        )
    )
    parser.add_argument(
        "--override_dir",
        type=Path,
        required=True,
        help="Directory containing <stem>.txt transcript override files.",
    )
    parser.add_argument(
        "--hashes",
        type=Path,
        required=True,
        help=(
            "JSON hash file produced by find_unchanged_overrides.py hash. "
            "It must contain entries for the split audio files, e.g. 10_0012.flac."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory containing the generated Whisper .txt files to overwrite.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report planned override copies without writing files.",
    )
    parser.add_argument("audio_files", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_hashes: dict[str, str] = json.loads(args.hashes.read_text(encoding="utf-8"))

    applied = 0
    skipped_no_override = 0
    skipped_no_hash = 0
    skipped_hash_mismatch = 0
    missing_transcripts = 0

    for audio_path in args.audio_files:
        stem = audio_path.stem
        override_path = args.override_dir / f"{stem}.txt"
        if not override_path.exists():
            skipped_no_override += 1
            continue

        hash_key = audio_path.name
        expected_hash = expected_hashes.get(hash_key)
        if expected_hash is None:
            print(f"SKIP override {stem}: no hash entry for {hash_key}")
            skipped_no_hash += 1
            continue

        current_hash = audio_hash(audio_path)
        if current_hash != expected_hash:
            print(
                f"SKIP override {stem}: audio hash mismatch "
                f"(current={current_hash}, expected={expected_hash})"
            )
            skipped_hash_mismatch += 1
            continue

        target_path = args.output_dir / f"{stem}.txt"
        if not target_path.exists():
            print(f"SKIP override {stem}: transcript not found: {target_path}")
            missing_transcripts += 1
            continue

        if args.dry_run:
            print(f"DRY RUN override {target_path} <- {override_path}")
        else:
            shutil.copyfile(override_path, target_path)
            print(f"APPLIED override {target_path} <- {override_path}")
        applied += 1

    print(
        "transcription overrides: "
        f"applied={applied} "
        f"skipped_no_override={skipped_no_override} "
        f"skipped_no_hash={skipped_no_hash} "
        f"skipped_hash_mismatch={skipped_hash_mismatch} "
        f"missing_transcripts={missing_transcripts}"
    )

    return 1 if missing_transcripts else 0


if __name__ == "__main__":
    raise SystemExit(main())
