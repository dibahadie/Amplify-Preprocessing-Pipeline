import argparse
from pathlib import Path


def extract_text_from_lab(lab_path, sep=" "):
    """Extracts and concatenates text from '<start> <end> <text>' lines."""
    segments = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split into [start_time, end_time, text]
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and parts[2].strip():
                segments.append(parts[2].strip())
    return sep.join(segments)


def convert_word_labs_to_txt(root_dir, dry_run=False, skip_existing=False, sep=" "):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")

    lab_files = sorted(
        p for p in root.rglob("*.word.lab") if p.is_file() and p.parent.name == "auto_word"
    )

    if not lab_files:
        print(f"No auto_word/*.word.lab files found under {root}")
        return

    created = 0
    skipped = 0

    for lab_file in lab_files:
        base_name = lab_file.name[: -len(".word.lab")]
        target_auto_txt = lab_file.parent.parent / "auto_txt"
        target_txt_file = target_auto_txt / f"{base_name}.txt"

        if skip_existing and target_txt_file.exists():
            skipped += 1
            if dry_run:
                print(f"[DRY RUN - SKIPPED] Already exists: {target_txt_file}")
            continue

        full_text = extract_text_from_lab(lab_file, sep=sep)

        if dry_run:
            preview = full_text[:80] + ("..." if len(full_text) > 80 else "")
            print(f"[DRY RUN] Would create: {target_txt_file} | Text: {preview!r}")
        else:
            target_auto_txt.mkdir(parents=True, exist_ok=True)
            with open(target_txt_file, "w", encoding="utf-8") as out_f:
                out_f.write(full_text + ("\n" if full_text else ""))
        created += 1

    status = "(DRY RUN - no files written)" if dry_run else "(Files written)"
    print(f"\nDone. Processed: {created}, Skipped: {skipped} {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate timestamped text from auto_word/*.word.lab into auto_txt/*.txt."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview output files without writing to disk.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite existing .txt files in auto_txt.",
    )
    parser.add_argument(
        "--sep",
        default=" ",
        help="Separator used when joining segments (default: single space).",
    )

    args = parser.parse_args()
    convert_word_labs_to_txt(
        args.root,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        sep=args.sep,
    )