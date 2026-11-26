import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def time_to_seconds(time_str):
    """Converts MM:SS.mmm string to total seconds (e.g., 01:03.500 -> 63.500)."""
    minutes, seconds = time_str.split(":")
    total_seconds = int(minutes) * 60 + float(seconds)
    return f"{total_seconds:.3f}"


def parse_log_for_transcripts(log_path):
    """Parses the log file to extract language mappings and timing/text for each chunk."""
    # Matches:
    #   Processing Song ID: 11640 (French / F)
    #   Processing Song ID: 11640 (F)
    #   Processing Song ID: 11640
    song_regex = re.compile(
        r"Processing Song ID:\s*(\d+)(?:\s*\((?:([^/)]+?)\s*/\s*)?([FM])\))?"
    )
    file_regex = re.compile(r"File: '.*?/(\d+_\d+)\.flac")
    lang_regex = re.compile(r"Detected language '([^']+)'")
    transcript_regex = re.compile(
        r"^\[(\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}\.\d{3})\]\s*(.*)"
    )

    dataset_info = {}
    current_key = None
    current_chunk = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            song_match = song_regex.search(line)
            if song_match:
                current_song_id = song_match.group(1)
                current_gender = (
                    song_match.group(3).strip() if song_match.group(3) else None
                )
                current_key = (current_song_id, current_gender)
                current_chunk = None

                if current_key not in dataset_info:
                    dataset_info[current_key] = {"chunks": {}}
                continue

            file_match = file_regex.search(line)
            if file_match and current_key:
                current_chunk = file_match.group(1)
                if current_chunk not in dataset_info[current_key]["chunks"]:
                    dataset_info[current_key]["chunks"][current_chunk] = {
                        "lang": None,
                        "lines": [],
                    }
                continue

            lang_match = lang_regex.search(line)
            if lang_match and current_key and current_chunk:
                dataset_info[current_key]["chunks"][current_chunk]["lang"] = (
                    lang_match.group(1).strip()
                )
                continue

            transcript_match = transcript_regex.search(line)
            if transcript_match and current_key and current_chunk:
                start_time = time_to_seconds(transcript_match.group(1))
                end_time = time_to_seconds(transcript_match.group(2))
                text = transcript_match.group(3).strip()

                dataset_info[current_key]["chunks"][current_chunk]["lines"].append(
                    f"{start_time} {end_time} {text}\n"
                )

    return dataset_info


def index_song_directories(root, target_song_ids):
    """Scans root recursively to locate each song_id directory regardless of nesting."""
    found = defaultdict(list)
    for dirpath, dirnames, _ in os.walk(root):
        matched = [d for d in dirnames if d in target_song_ids]
        for d in matched:
            found[d].append(Path(dirpath) / d)
            dirnames.remove(d)
    return found


def resolve_target_dir(root, candidates, detected_lang, gender, song_id, new_chunk_base):
    """Finds the matching song directory containing the target .flac file."""
    # 1. Check the standard reorganized paths first
    preferred = []
    if gender:
        preferred.append(root / detected_lang / gender / song_id)
    preferred.append(root / detected_lang / song_id)

    for path in preferred:
        if (path / f"{new_chunk_base}.flac").exists():
            return path

    # 2. Fall back to any discovered directory for this song_id matching detected_lang/gender
    for path in candidates:
        parts = path.parts
        lang_ok = detected_lang in parts
        gender_ok = (gender is None) or (gender in parts)
        if lang_ok and gender_ok and (path / f"{new_chunk_base}.flac").exists():
            return path

    # 3. Last resort: any discovered directory for this song_id containing the .flac
    for path in candidates:
        if (path / f"{new_chunk_base}.flac").exists():
            return path

    return preferred[0]


def generate_word_labels(root_dir, log_path, dry_run=False):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")

    dataset_info = parse_log_for_transcripts(log_path)
    song_ids = {song_id for song_id, _ in dataset_info.keys()}
    song_dir_index = index_song_directories(root, song_ids)

    counters = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    for (song_id, gender), info in dataset_info.items():
        candidates = song_dir_index.get(song_id, [])

        # If gender wasn't in the log, try inferring it from a discovered parent folder (F or M)
        effective_gender = gender
        if effective_gender is None:
            for path in candidates:
                if path.parent.name in {"F", "M"}:
                    effective_gender = path.parent.name
                    break

        gender_key = effective_gender or "_none_"

        for chunk_base, chunk_data in info["chunks"].items():
            detected_lang = chunk_data["lang"]
            if not detected_lang:
                continue

            n_prefix, _ = chunk_base.split("_")

            counters[detected_lang][gender_key][song_id][n_prefix] += 1
            new_m = counters[detected_lang][gender_key][song_id][n_prefix]
            new_chunk_base = f"{n_prefix}_{new_m:04d}"

            target_song_dir = resolve_target_dir(
                root,
                candidates,
                detected_lang,
                effective_gender,
                song_id,
                new_chunk_base,
            )
            target_flac = target_song_dir / f"{new_chunk_base}.flac"
            target_auto_word = target_song_dir / "auto_word"
            target_lab_file = target_auto_word / f"{new_chunk_base}.word.lab"

            if not target_flac.exists():
                if dry_run:
                    print(f"[DRY RUN - SKIPPED] Audio missing: {target_flac}")
                continue

            if dry_run:
                print(
                    f"[DRY RUN] Would create: {target_lab_file} "
                    f"({len(chunk_data['lines'])} lines)"
                )
            else:
                target_auto_word.mkdir(parents=True, exist_ok=True)
                with open(target_lab_file, "w", encoding="utf-8") as lab_f:
                    lab_f.writelines(chunk_data["lines"])

    print(
        f"\nLabel generation complete. "
        f"{'(DRY RUN - no files written)' if dry_run else ''}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate .word.lab files from transcription logs."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the main reorganized dataset directory.",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to the original transcription log text file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the file creations without modifying the disk.",
    )

    args = parser.parse_args()
    generate_word_labels(args.root, args.log, args.dry_run)