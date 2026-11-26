import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Parses the log file to extract language mappings for each chunk."""
    # Matches:
    #   Processing Song ID: 11640 (French / F)
    #   Processing Song ID: 11640 (F)
    #   Processing Song ID: 11640
    song_regex = re.compile(
        r"Processing Song ID:\s*(\d+)(?:\s*\((?:([^/)]+?)\s*/\s*)?([FM])\))?"
    )
    file_regex = re.compile(r"File: '.*?/(\d+_\d+)\.flac")
    lang_regex = re.compile(r"Detected language '([^']+)'")

    dataset_info = {}
    current_song_id = None
    current_chunk = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            song_match = song_regex.search(line)
            if song_match:
                current_song_id = song_match.group(1)
                original_lang = song_match.group(2).strip() if song_match.group(2) else None
                gender = song_match.group(3).strip() if song_match.group(3) else None

                dataset_info[current_song_id] = {
                    "original_lang": original_lang,
                    "gender": gender,
                    "chunks": {},
                }
                continue

            file_match = file_regex.search(line)
            if file_match and current_song_id:
                current_chunk = file_match.group(1)
                continue

            lang_match = lang_regex.search(line)
            if lang_match and current_song_id and current_chunk:
                detected_lang = lang_match.group(1).strip()
                dataset_info[current_song_id]["chunks"][current_chunk] = detected_lang
                current_chunk = None

    return dataset_info


def index_song_directories(root, target_song_ids):
    """Scans root recursively to locate each song_id directory regardless of nesting."""
    found = defaultdict(list)
    for dirpath, dirnames, _ in os.walk(root):
        matched = [d for d in dirnames if d in target_song_ids]
        for d in matched:
            found[d].append(Path(dirpath) / d)
            # Don't descend into matched song directories (e.g., auto_txt)
            dirnames.remove(d)
    return found


def resolve_source_dir(candidates, original_lang, gender):
    """Picks the best matching directory if a song_id exists in multiple places."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for path in candidates:
        parts = path.parts
        lang_ok = (original_lang is None) or (original_lang in parts)
        gender_ok = (gender is None) or (gender in parts)
        if lang_ok and gender_ok:
            return path
    return candidates[0]


def process_dataset(root_dir, log_path, dry_run=False):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")

    dataset_info = parse_log_file(log_path)
    song_dir_index = index_song_directories(root, set(dataset_info.keys()))

    # counters[lang][gender_key][song_id][n_prefix] = next_m
    counters = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    for song_id, info in dataset_info.items():
        original_lang = info["original_lang"]
        gender = info["gender"]

        source_song_dir = resolve_source_dir(
            song_dir_index.get(song_id, []), original_lang, gender
        )
        if source_song_dir is None:
            print(f"Warning: Song directory '{song_id}' not found anywhere under {root}")
            continue

        # If gender wasn't in the log, try inferring it from the parent folder (F or M)
        if gender is None and source_song_dir.parent.name in {"F", "M"}:
            gender = source_song_dir.parent.name

        gender_key = gender or "_none_"

        for chunk_base, detected_lang in info["chunks"].items():
            n_prefix, _ = chunk_base.split("_")

            if gender:
                target_song_dir = root / detected_lang / gender / song_id
            else:
                target_song_dir = root / detected_lang / song_id
            target_auto_txt = target_song_dir / "auto_txt"

            counters[detected_lang][gender_key][song_id][n_prefix] += 1
            new_m = counters[detected_lang][gender_key][song_id][n_prefix]
            new_chunk_base = f"{n_prefix}_{new_m:04d}"

            if not dry_run:
                target_song_dir.mkdir(parents=True, exist_ok=True)
                target_auto_txt.mkdir(parents=True, exist_ok=True)

            # Move all related files in the main song directory (.flac, .json, .vad.lab, etc.)
            for file_path in source_song_dir.glob(f"{chunk_base}.*"):
                if file_path.is_file():
                    extension = file_path.name[len(chunk_base):]
                    target_file = target_song_dir / f"{new_chunk_base}{extension}"

                    if file_path.resolve() == target_file.resolve():
                        continue

                    if dry_run:
                        print(f"[DRY RUN] Move: {file_path} -> {target_file}")
                    else:
                        shutil.move(str(file_path), str(target_file))

            # Move the corresponding auto_txt file
            source_txt_file = source_song_dir / "auto_txt" / f"{chunk_base}.txt"
            if source_txt_file.exists():
                target_txt_file = target_auto_txt / f"{new_chunk_base}.txt"
                if source_txt_file.resolve() == target_txt_file.resolve():
                    continue

                if dry_run:
                    print(f"[DRY RUN] Move: {source_txt_file} -> {target_txt_file}")
                else:
                    shutil.move(str(source_txt_file), str(target_txt_file))

    print(
        f"\nProcessing complete. "
        f"{'(DRY RUN - no files moved)' if dry_run else '(Files successfully reorganized)'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize audio dataset by detected language logs."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the main dataset directory.",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to the transcription log text file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the move operations without modifying any files.",
    )

    args = parser.parse_args()
    process_dataset(args.root, args.log, args.dry_run)