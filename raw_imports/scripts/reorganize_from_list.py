import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def is_subpath(path, parent):
    """Returns True if path is inside parent (or equal to parent)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def get_gender_from_path(path):
    """Returns 'F' or 'M' if the song directory's parent is a gender folder."""
    parent_name = path.parent.parent.name
    return parent_name if parent_name in {"F", "M"} else None


def scan_source_files(root, excluded_dirs=(), include_filtered=False):
    """
    Snapshots all source files under root and indexes .flac files by
    (effective_song_id, chunk_base).
    If a song_id exists in both F and M folders, it is indexed as:
      - M: <song_id>00
      - F: <song_id>11
    """
    filtered_dir = root / "filtered"
    all_files = []
    raw_flacs = []
    song_genders = defaultdict(set)

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(is_subpath(path, ex) for ex in excluded_dirs):
            continue
        if not include_filtered and is_subpath(path, filtered_dir):
            continue

        all_files.append(path)
        if path.suffix.lower() == ".flac":
            song_id = path.parent.name
            gender = get_gender_from_path(path)
            if gender:
                song_genders[song_id].add(gender)
            raw_flacs.append((path, song_id, gender))

    overlapping_ids = {
        song_id for song_id, genders in song_genders.items() if {"F", "M"}.issubset(genders)
    }

    flac_index = defaultdict(list)
    for path, song_id, gender in raw_flacs:
        if song_id in overlapping_ids:
            effective_id = f"{song_id}00" if gender == "M" else f"{song_id}11"
        else:
            effective_id = song_id
        flac_index[(effective_id, path.stem)].append(path)

    return all_files, flac_index, overlapping_ids


def find_segment_source(flac_index, rel_path, handled_files):
    """Finds the best unclaimed source .flac for a single segment path."""
    song_id = rel_path.parent.name
    chunk_base = rel_path.stem
    target_gender = get_gender_from_path(rel_path)

    keys_to_try = [(song_id, chunk_base)]

    # If list has <id>00 (M) or <id>11 (F), also check base <id>
    if target_gender == "M" and song_id.endswith("00"):
        keys_to_try.append((song_id[:-2], chunk_base))
    elif target_gender == "F" and song_id.endswith("11"):
        keys_to_try.append((song_id[:-2], chunk_base))

    # Reverse fallback: if list has base <id>, also check <id>00 / <id>11
    if target_gender == "M":
        keys_to_try.append((f"{song_id}00", chunk_base))
    elif target_gender == "F":
        keys_to_try.append((f"{song_id}11", chunk_base))
    else:
        keys_to_try.append((f"{song_id}00", chunk_base))
        keys_to_try.append((f"{song_id}11", chunk_base))

    seen = set()
    candidates = []
    for key in keys_to_try:
        for path in flac_index.get(key, []):
            resolved = path.resolve()
            if resolved in seen or resolved in handled_files or not path.exists():
                continue
            seen.add(resolved)
            candidates.append(path)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    target_parts = set(rel_path.parts)
    return max(candidates, key=lambda p: len(set(p.parts) & target_parts))


def collect_companion_files(song_dir, chunk_base):
    """Returns all existing files belonging to a single segment in song_dir."""
    files = []

    for file_path in song_dir.glob(f"{chunk_base}.*"):
        if file_path.is_file():
            files.append((file_path, Path(file_path.name)))

    txt_file = song_dir / "auto_txt" / f"{chunk_base}.txt"
    if txt_file.is_file():
        files.append((txt_file, Path("auto_txt") / f"{chunk_base}.txt"))

    word_file = song_dir / "auto_word" / f"{chunk_base}.word.lab"
    if word_file.is_file():
        files.append((word_file, Path("auto_word") / f"{chunk_base}.word.lab"))

    return files


def copy_file(src, dst, dry_run=False):
    if src.resolve() == dst.resolve():
        return "skipped"

    if dst.exists():
        print(f"Warning: Destination exists, skipping to prevent overwrite: {src} -> {dst}")
        return "collision"

    if dry_run:
        print(f"[DRY RUN] Copy: {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    return "copied"


def reorganize_from_list(
    root_dir,
    list_path,
    dest_dir=None,
    dry_run=False,
    include_filtered=False,
):
    source_root = Path(root_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {source_root.resolve()}")

    dest_root = Path(dest_dir) if dest_dir else source_root
    same_root = source_root.resolve() == dest_root.resolve()
    filtered_root = dest_root / "filtered"

    # If --dest is nested inside --root, exclude it when scanning source files
    excluded_dirs = []
    if not same_root and is_subpath(dest_root, source_root):
        excluded_dirs.append(dest_root)

    all_source_files, flac_index, overlapping_ids = scan_source_files(
        source_root,
        excluded_dirs=excluded_dirs,
        include_filtered=include_filtered,
    )
    if overlapping_ids:
        print(f"Detected {len(overlapping_ids)} overlapping Song_ID(s) across F and M.")

    handled_files = set()
    planned_targets = set()

    copied_chunks = 0
    missing_chunks = 0
    skipped_chunks = 0
    collision_files = 0

    # Process the input list strictly one line (segment) at a time
    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\\", "/")
            if not line or line.startswith("#"):
                continue

            rel_path = Path(line)
            if rel_path.suffix.lower() != ".flac":
                continue

            chunk_base = rel_path.stem
            target_song_dir = dest_root / rel_path.parent
            target_flac = target_song_dir / rel_path.name
            target_flac_res = target_flac.resolve()

            # Skip duplicate lines in the list
            if target_flac_res in planned_targets:
                skipped_chunks += 1
                continue

            # If reorganizing inside --root and file is already at target, keep it in place
            if same_root and target_flac.is_file():
                for src, _ in collect_companion_files(target_song_dir, chunk_base):
                    handled_files.add(src.resolve())
                planned_targets.add(target_flac_res)
                skipped_chunks += 1
                continue

            # Find the source segment and copy it immediately
            source_flac = find_segment_source(flac_index, rel_path, handled_files)
            if source_flac is None:
                print(f"Warning: Source not found for {rel_path}")
                missing_chunks += 1
                continue

            source_song_dir = source_flac.parent
            companions = collect_companion_files(source_song_dir, chunk_base)

            for src, rel_subpath in companions:
                handled_files.add(src.resolve())
                dst = target_song_dir / rel_subpath
                result = copy_file(src, dst, dry_run=dry_run)
                if result == "collision":
                    collision_files += 1

            planned_targets.add(target_flac_res)
            copied_chunks += 1

    # Copy all remaining unlisted files into <dest_root>/filtered/ keeping original structure
    source_filtered_dir = source_root / "filtered"
    filtered_files = 0
    for src_file in all_source_files:
        if src_file.resolve() in handled_files:
            continue
        if is_subpath(src_file, source_filtered_dir):
            continue

        rel_orig = src_file.relative_to(source_root)
        dst_file = filtered_root / rel_orig
        result = copy_file(src_file, dst_file, dry_run=dry_run)
        if result == "copied":
            filtered_files += 1
        elif result == "collision":
            collision_files += 1

    status = "(DRY RUN - no files copied)" if dry_run else ""
    print(
        f"\nDone. Copied segments: {copied_chunks}, Already in place / duplicate: {skipped_chunks}, "
        f"Missing: {missing_chunks}, Collisions skipped: {collision_files}, "
        f"Filtered files copied: {filtered_files} {status}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy dataset files segment-by-segment into the structure defined by a target path list."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the current dataset root directory.",
    )
    parser.add_argument(
        "--list",
        required=True,
        help="Path to the text file containing desired relative .flac paths.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Output root directory (recommended so --root stays clean and untouched).",
    )
    parser.add_argument(
        "--include-filtered",
        action="store_true",
        help="Also search inside <root>/filtered/ when locating source files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview copy operations without writing any files.",
    )

    args = parser.parse_args()
    reorganize_from_list(
        args.root,
        args.list,
        dest_dir=args.dest,
        dry_run=args.dry_run,
        include_filtered=args.include_filtered,
    )