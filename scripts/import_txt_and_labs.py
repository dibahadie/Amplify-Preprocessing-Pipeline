#!/usr/bin/env python3
from __future__ import annotations

from typing import cast

import argparse
import re
import sys
from pathlib import Path
import fileio.speechdb.db_names as dbn
from fileio import lab
import yaml
from tqdm import tqdm

SRCBASE="sources.yml"
def find_source_file(file_or_dir:Path):
    dirbase = file_or_dir if file_or_dir.is_dir() else file_or_dir.parent
    while True:
        if (dirbase/SRCBASE).exists():
            return dirbase/SRCBASE
        if dirbase == dirbase.parent:
            break
        dirbase = dirbase.parent
    raise FileNotFoundError(f"Could not find a sources.yml for file {file_or_dir}")

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_SRC = Path.home() / 'pysrc'
if str(PY_SRC) not in sys.path:
    sys.path.insert(0, str(PY_SRC))

# Vowel source labels follow the eSpeak-ng French notation used during raw
# import. Remap them here into the target XSAMPA-like inventory exported by
# this repository.
VOWEL_LABEL_MAP = {
    "I": "i",
    "Y": "2",
    "W": "9",
    "E~": "e~",
    "A~": "a~",
    "O~": "o~",
    "W~": "9~",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f'Import txt and phoneme labels referenced by ROOT/{SRCBASE}. '
            'Vowel entries additionally import their source .lab files, while '
            'song entries get virtual .lab files generated from their phoneme sequences.'
        )
    )
    parser.add_argument(
        'db_config',
        nargs='?',
        default=str(REPO_ROOT),
        help='Database root or db_config.py path. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--tag_txt_out',
        default='txt',
        help='speechdb output tag for imported txt files. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--tag_phn_out',
        default='phon_xsampa',
        help='speechdb output tag for imported xampa lab files. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--tag_lab_out',
        default='lab_phon',
        help=(
            'speechdb output tag for source or generated .lab files. '
            'Vowel entries copy their source .lab files; song entries generate virtual '
            '.lab files from the phoneme sequence with -1 start and end times. '
            '(Def.: %(default)s)'
        ),
    )
    parser.add_argument(
        '--tag_vad_out',
        default='vad_loud',
        help='speechdb output tag for imported VAD files. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--speaker',
        action='append',
        default=[],
        help='Only process matching speaker directory names, e.g. CP.',
    )
    parser.add_argument(
        '--beg_ind',
        type=int,
        default=0,
        help='Minimum phrase index to include. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--end_ind',
        type=int,
        default=None,
        help='Maximum phrase index to include.',
    )
    parser.add_argument(
        '--max_search_fac',
        default=10,
        type=int,
        help='Maximum factor used by speechdb when searching sparse indices. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Report planned work without writing output files.',
    )
    parser.add_argument(
        '--preserve',
        action='store_true',
        help='Do not overwrite existing output files. (Def.: %(default)s)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print one line per processed file in addition to the progress bar.',
    )
    parser.add_argument(
        '--annotation_style',
        choices=['vax', 'whisper'],
        default='vax',
        help=(
            'Annotation file layout for songs. '
            '"vax": final_txt/<seg>.vax.txt + final_phon/<seg>.lphn (default). '
            '"whisper": <seg>.final.txt + <seg>.final.phn (legacy).'
        ),
    )
    return parser.parse_args()


def normalize_repo_path(path_str: str) -> Path:
    cleaned = path_str
    if cleaned.startswith('.//'):
        cleaned = cleaned[3:]
    elif cleaned.startswith('./'):
        cleaned = cleaned[2:]
    return (REPO_ROOT / cleaned).resolve()


def is_vowel_source(ori_path: str) -> bool:
    return "vowel" in ori_path.lower()


def source_lab_path_from_ori(ori_path: str) -> Path:
    ori = normalize_repo_path(ori_path)
    source_lab = ori.with_suffix(".lab")
    if not source_lab.exists():
        raise FileNotFoundError(f"Could not find lab file for vowel sound {ori_path}")
    return source_lab


def map_vowel_labels(labels: list[str | None]) -> list[str | None]:
    return [VOWEL_LABEL_MAP.get(label, label) if label is not None else None for label in labels]


def load_vowel_lab_from_ori(ori_path: str) -> tuple[list[float], list[float], list[str | None]]:
    from fileio.lab import load as load_lab

    starts, ends, labels = load_lab(source_lab_path_from_ori(ori_path))
    return list(starts), list(ends), map_vowel_labels(labels)


def vowel_label_text_from_ori(ori_path: str) -> str:
    _, _, labels = load_vowel_lab_from_ori(ori_path)
    return " ".join(label for label in labels if label is not None) + "\n"


def source_phn_path_from_ori(ori_path: str, annotation_style: str = "vax") -> Path:
    ori = normalize_repo_path(ori_path)
    if annotation_style == "vax":
        source_phn = ori.parent / "final_phon" / (ori.stem + ".vax.lphn")
    else:
        source_phn = ori.with_suffix(".final.phn")
    if not source_phn.exists():
        raise FileNotFoundError(f"Could not find phoneme file for sound {ori_path}")
    return source_phn


def source_txt_path_from_ori(ori_path: str, annotation_style: str = "vax") -> Path:
    ori = normalize_repo_path(ori_path)
    if annotation_style == "vax":
        source_txt = ori.parent / "final_txt" / (ori.stem + ".vax.txt")
    else:
        source_txt = ori.with_suffix(".final.txt")
    if not source_txt.exists():
        raise FileNotFoundError(f"Could not find text file for sound {ori_path}")
    return source_txt


def source_vad_path_from_ori(ori_path: str) -> Path:
    ori = normalize_repo_path(ori_path)
    source_vad = ori.with_suffix(".vad.lab")
    if not source_vad.exists():
        raise FileNotFoundError(f"Could not find split VAD file for sound {ori_path}")
    return source_vad


def source_vad_json_path_from_ori(ori_path: str) -> Path | None:
    source_vad = source_vad_path_from_ori(ori_path)
    source_vad_json = Path(str(source_vad) + ".json")
    if source_vad_json.exists():
        return source_vad_json
    return None


def source_content_from_ori(ori_path: str, typ: str, annotation_style: str = "vax") -> str:
    if typ == "lab":
        return source_lab_path_from_ori(ori_path).read_text(encoding="utf-8")
    if is_vowel_source(ori_path):
        return vowel_label_text_from_ori(ori_path)
    if typ == "phn":
        return source_phn_path_from_ori(ori_path, annotation_style).read_text(encoding="utf-8")
    return source_txt_path_from_ori(ori_path, annotation_style).read_text(encoding="utf-8")


def song_virtual_lab_from_ori(ori_path: str, annotation_style: str = "vax") -> tuple[list[float], list[float], list[str]]:
    labels = source_phn_path_from_ori(ori_path, annotation_style).read_text(encoding="utf-8").split()
    starts = [-1.0] * len(labels)
    ends = [-1.0] * len(labels)
    return starts, ends, labels


def clip_to_segment(
    starts: list[float],
    ends: list[float],
    labels: list[str],
    seg_start: float,
    seg_end: float,
) -> tuple[list[float], list[float], list[str]]:
    clipped_starts: list[float] = []
    clipped_ends: list[float] = []
    clipped_labels: list[str] = []

    for start, end, label in zip(starts, ends, labels):
        overlap_start = max(start, seg_start)
        overlap_end = min(end, seg_end)
        if overlap_end <= overlap_start:
            continue
        clipped_starts.append(overlap_start - seg_start)
        clipped_ends.append(overlap_end - seg_start)
        clipped_labels.append(label)

    return clipped_starts, clipped_ends, clipped_labels


def make_dir_filter(speakers: list[str]):
    if not speakers:
        return None
    escaped = '|'.join(re.escape(speaker) for speaker in speakers)
    return re.compile(rf'.*/(?:{escaped})$')


def load_source_mapping(sources_path: Path) -> dict[Path, dict]:
    with sources_path.open('r', encoding='utf-8') as handle:
        entries = yaml.safe_load(handle)

    mapping: dict[Path, dict] = {}
    for entry in entries:
        wav_path = normalize_repo_path(entry['FILE'])
        mapping[wav_path] = entry
    return mapping


def build_entries(args: argparse.Namespace) -> list[dict]:
    entries: list[dict] = []
    current_config = None
    dir_filter = make_dir_filter(args.speaker)
    mapping: dict[Path, dict] = {}
    current_src = None
    for item in dbn.generate_agamemnon_db_entries_from_tags(
        args.db_config,
        tags=['wav'],
        beg_ind=args.beg_ind,
        end_ind=args.end_ind,
        error_on_missing=False,
        return_structure=True,
        dir_filter_pat=dir_filter,
        max_search_fac=args.max_search_fac,
    ):
        if 'base' in item:
            current_config = dbn.retrieve_db_config(cast(str, item['base']))
            new_src = find_source_file(Path(cast(str, item['base'])))
            if new_src != current_src:
                current_src = new_src
                mapping = load_source_mapping(current_src)
            continue

        if current_config is None:
            raise RuntimeError('speechdb iterator returned a file entry before providing its db_config')

        wav_path = Path(cast(str, item['wav'])).resolve()
        try:
            source_entry = mapping[wav_path]
        except KeyError as exc:
            raise KeyError(f'no entry found for audio file {wav_path} in {current_src}') from exc

        entries.append(
            {
                'wav_path': wav_path,
                'source_entry': source_entry,
                'target_txt': Path(dbn.get_filename_for_tag(current_config, args.tag_txt_out, cast(int, item['ind']))),
                'target_phn': Path(dbn.get_filename_for_tag(current_config, args.tag_phn_out, cast(int, item['ind']))),
                'target_lab': Path(dbn.get_filename_for_tag(current_config, args.tag_lab_out, cast(int, item['ind']))),
                'target_vad': Path(dbn.get_filename_for_tag(current_config, args.tag_vad_out, cast(int, item['ind']))),
                'ind': cast(int, item['ind']),
            }
        )
    return entries


def convert_entry(
    source_entry: dict,
    target_path: Path,
    overwrite: bool,
    dry_run: bool,
    typ="phn",
    annotation_style: str = "vax",
) -> tuple[str, Path]:
    result_tag = None
    if target_path.exists() and not overwrite:
        result_tag = 'skipped-existing'
    else:
        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if typ == "lab" and not is_vowel_source(source_entry['ORI']):
                starts, ends, labels = song_virtual_lab_from_ori(source_entry['ORI'], annotation_style)
                lab.store(str(target_path), starts, ends, labels)
            elif typ == "lab":
                starts, ends, labels = load_vowel_lab_from_ori(source_entry['ORI'])
                lab.store(str(target_path), starts, ends, labels)
            elif typ == "vad":
                source_vad_path = source_vad_path_from_ori(source_entry['ORI'])
                target_path.write_text(source_vad_path.read_text(encoding="utf-8"), encoding="utf-8")
                source_vad_json_path = source_vad_json_path_from_ori(source_entry['ORI'])
                if source_vad_json_path is not None:
                    vad_meta_text = source_vad_json_path.read_text(encoding="utf-8")
                    vad_meta_text = re.sub(
                        r'("outfile"\s*:\s*")[^"]*(")',
                        rf'\1{target_path}\2',
                        vad_meta_text,
                        count=1,
                    )
                    Path(str(target_path) + ".json").write_text(vad_meta_text, encoding="utf-8")
            else:
                cont = source_content_from_ori(source_entry['ORI'], typ, annotation_style)
                target_path.write_text(cont, encoding="utf-8")
            result_tag = 'written'
        else:
            result_tag = 'dry-run'

    return result_tag, target_path


def main() -> int:
    args = parse_args()
    entries = build_entries(args)

    txt_written = 0
    txt_skipped = 0
    txt_failed = 0
    phn_written = 0
    phn_skipped = 0
    phn_failed = 0
    lab_written = 0
    lab_skipped = 0
    lab_failed = 0
    vad_written = 0
    vad_skipped = 0
    vad_failed = 0
    failed = 0

    progress = tqdm(entries, desc='Importing txt/phn/lab/vad', unit='entry')
    for entry in progress:
        entry_failed = False

        try:
            txt_status, txt_out_path = convert_entry(
                source_entry=entry['source_entry'],
                target_path=entry['target_txt'],
                overwrite=not args.preserve,
                dry_run=args.dry_run,
                typ="txt",
                annotation_style=args.annotation_style,
            )
            if txt_status == 'skipped-existing':
                txt_skipped += 1
            else:
                txt_written += 1
            if args.verbose:
                tqdm.write(f'txt {txt_status:16s} {txt_out_path}')
        except Exception as exc:  # noqa: BLE001
            entry_failed = True
            txt_failed += 1
            tqdm.write(f"txt failed  {entry['wav_path']} :: {exc}")

        try:
            phn_status, phn_out_path = convert_entry(
                source_entry=entry['source_entry'],
                target_path=entry['target_phn'],
                overwrite=not args.preserve,
                dry_run=args.dry_run,
                typ="phn",
                annotation_style=args.annotation_style,
            )
            if phn_status == 'skipped-existing':
                phn_skipped += 1
            else:
                phn_written += 1
            if args.verbose:
                tqdm.write(f'phn {phn_status:16s} {phn_out_path}')
        except Exception as exc:  # noqa: BLE001
            entry_failed = True
            phn_failed += 1
            tqdm.write(f"phn failed  {entry['wav_path']} :: {exc}")

        target_lab = entry['target_lab']
        if target_lab is not None:
            try:
                lab_status, lab_out_path = convert_entry(
                    source_entry=entry['source_entry'],
                    target_path=target_lab,
                    overwrite=not args.preserve,
                    dry_run=args.dry_run,
                    typ="lab",
                    annotation_style=args.annotation_style,
                )
                if lab_status == 'skipped-existing':
                    lab_skipped += 1
                else:
                    lab_written += 1
                if args.verbose:
                    tqdm.write(f'lab {lab_status:16s} {lab_out_path}')
            except Exception as exc:  # noqa: BLE001
                entry_failed = True
                lab_failed += 1
                tqdm.write(f"lab failed  {entry['wav_path']} :: {exc}")

        target_vad = entry['target_vad']
        if target_vad is not None:
            try:
                vad_status, vad_out_path = convert_entry(
                    source_entry=entry['source_entry'],
                    target_path=target_vad,
                    overwrite=not args.preserve,
                    dry_run=args.dry_run,
                    typ="vad",
                    annotation_style=args.annotation_style,
                )
                if vad_status == 'skipped-existing':
                    vad_skipped += 1
                else:
                    vad_written += 1
                if args.verbose:
                    tqdm.write(f'vad {vad_status:16s} {vad_out_path}')
            except Exception as exc:  # noqa: BLE001
                entry_failed = True
                vad_failed += 1
                tqdm.write(f"vad failed  {entry['wav_path']} :: {exc}")

        if entry_failed:
            failed += 1

        progress.set_postfix(
            txt_written=txt_written,
            txt_skipped=txt_skipped,
            txt_failed=txt_failed,
            phn_written=phn_written,
            phn_skipped=phn_skipped,
            phn_failed=phn_failed,
            lab_written=lab_written,
            lab_skipped=lab_skipped,
            lab_failed=lab_failed,
            vad_written=vad_written,
            vad_skipped=vad_skipped,
            vad_failed=vad_failed,
        )

    print(
        f'processed={len(entries)} failed_entries={failed}\n'
        f'txt: written_or_planned={txt_written} skipped_existing={txt_skipped} failed={txt_failed}\n'
        f'phn: written_or_planned={phn_written} skipped_existing={phn_skipped} failed={phn_failed}\n'
        f'lab: written_or_planned={lab_written} skipped_existing={lab_skipped} failed={lab_failed}\n'
        f'vad: written_or_planned={vad_written} skipped_existing={vad_skipped} failed={vad_failed}'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
