#! /usr/bin/env python
from __future__ import annotations
from typing import cast
from pathlib import Path

# Custom internal libraries for handling the database and audio slicing
import fileio.speechdb.db_names as dbn
import fileio.speechdb.db_names.db_parser as db_parser
from svp_cmds.cut_silences import cut_silences, group_segments_by_max_duration, concatenate_segments
from fileio import lab
from pysndfile import sndio

import argparse
import glob
import json
import re
import yaml

# These are the only labels allowed in a strictly formatted VAD file.
VALID_VAD_LABELS = {"SIL", "SIL_SPLIT", "SIL_NOSPLIT", "VOI"}
SILENCE_VAD_LABELS = {"SIL", "SIL_SPLIT", "SIL_NOSPLIT"}


def _duration_gt1(value: str) -> float:
    """Helper function to make sure users don't ask for a max audio length less than 1 second."""
    v = float(value)
    if v <= 1:
        raise argparse.ArgumentTypeError(f"must be > 1, got {v}")
    return v


def load_validated_vad_lab(vad_path: str | Path) -> tuple[list[float], list[float], list[str]]:
    """
    Reads the Voice Activity Detection (VAD) file.
    This acts as a security guard: if it sees a label that isn't 'SIL' or 'VOI', 
    it crashes the script to prevent garbage data from entering your AI training pipeline.
    """
    path = str(vad_path)
    starts_raw, ends_raw, labels_raw = lab.load(path, force_3_tuples=True)
    labels: list[str] = []
    
    # Check every single label against our VALID_VAD_LABELS list
    for idx, label in enumerate(labels_raw, start=1):
        label_str = cast(str, label).strip().upper()
        if label_str not in VALID_VAD_LABELS:
            raise ValueError(
                f"invalid VAD label in {path}: entry {idx} must be one of "
                f"{sorted(VALID_VAD_LABELS)}, got {label_str!r}"
            )
        labels.append(label_str)

    return list(starts_raw), list(ends_raw), labels


def load_generic_lab(lab_path: str | Path) -> tuple[list[float], list[float], list[str]]:
    """
    Reads Word and Phoneme files.
    Unlike the VAD loader above, this is "Generic" and has no security checks.
    It will blindly accept any text (like Chinese characters or phonemes) without crashing.
    """
    path = str(lab_path)
    starts_raw, ends_raw, labels_raw = lab.load(path, force_3_tuples=True)
    labels = [cast(str, label).strip() for label in labels_raw]
    return list(starts_raw), list(ends_raw), labels


def merge_adjacent_segments(
    starts: list[float],
    ends: list[float],
    labels: list[str],
) -> tuple[list[float], list[float], list[str]]:
    """
    Cleans up messy timelines. 
    If the slicer creates two "Silence" blocks right next to each other 
    (e.g., [0.0 to 1.0 SIL] and [1.0 to 2.0 SIL]), this function merges them 
    into a single block ([0.0 to 2.0 SIL]).
    """
    if not starts:
        return [], [], []

    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    merged_labels = [labels[0]]
    
    for start, end, label in zip(starts[1:], ends[1:], labels[1:]):
        # If the label is the same as the last one, and they touch exactly (no time gap)
        if merged_labels[-1] == label and abs(merged_ends[-1] - start) < 1e-9:
            merged_ends[-1] = end # Extend the end time of the previous label
        else:
            merged_starts.append(start)
            merged_ends.append(end)
            merged_labels.append(label)
            
    return merged_starts, merged_ends, merged_labels


def clip_vad_to_segments(
    vad_starts: list[float],
    vad_ends: list[float],
    vad_labels: list[str],
    snd_segs: list[dict],
) -> tuple[list[float], list[float], list[str]]:
    """
    THE TIMELINE CALCULATOR - Updated to safely handle -1 placeholder timestamps.
    """
    out_starts: list[float] = []
    out_ends: list[float] = []
    out_labels: list[str] = []

    for seg_info in snd_segs:
        seg_ori_start = float(seg_info['seg_ori_start_s'])
        seg_ori_end = float(seg_info['seg_ori_end_s'])
        seg_chunk_start = float(seg_info.get('seg_chunk_start_s', 0.0))
        
        for vs, ve, vl in zip(vad_starts, vad_ends, vad_labels):
            
            # Handle the start timestamp safely
            if vs == -1:
                cs = -1.0
                out_s = -1.0
            else:
                cs = max(vs, seg_ori_start)
                out_s = cs - seg_ori_start + seg_chunk_start

            # Handle the end timestamp safely
            if ve == -1:
                ce = -1.0
                out_e = -1.0
            else:
                ce = min(ve, seg_ori_end)
                out_e = ce - seg_ori_start + seg_chunk_start
            
            # If both are normal timestamps, check if they fall outside this chunk
            if cs != -1 and ce != -1 and ce <= cs:
                continue
                
            # Filter out entries where the valid timestamp is completely outside the chunk
            if (cs != -1 and cs >= seg_ori_end) or (ce != -1 and ce <= seg_ori_start):
                continue
                
            out_starts.append(out_s)
            out_ends.append(out_e)
            out_labels.append(vl)

    return out_starts, out_ends, out_labels


def write_split_vad(
    stem_path: Path,
    snd_segs: list[dict],
    vad_starts: list[float],
    vad_ends: list[float],
    vad_labels: list[str],
    source_vad_meta: dict,
    source_vad_path: str,
) -> None:
    """
    Uses the calculator above to generate the new timestamps for the VAD file, 
    saves the new `.vad.lab` file, AND saves a `.json` metadata file next to it.
    """
    split_starts, split_ends, split_labels = clip_vad_to_segments(
        vad_starts=vad_starts,
        vad_ends=vad_ends,
        vad_labels=vad_labels,
        snd_segs=snd_segs,
    )
    vad_lab_path = Path(str(stem_path) + ".vad.lab")
    lab.store(str(vad_lab_path), split_starts, split_ends, split_labels)

    # VADs require a JSON metadata sidecar file for downstream processing
    vad_meta = dict(source_vad_meta)
    vad_meta["outfile"] = str(vad_lab_path)
    vad_meta["source_vad_file"] = source_vad_path
    vad_meta["source_segments"] = snd_segs
    Path(str(vad_lab_path) + ".json").write_text(
        json.dumps(vad_meta, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_split_generic_lab(
    stem_path: Path,
    snd_segs: list[dict],
    starts: list[float],
    ends: list[float],
    labels: list[str],
    ext: str
) -> None:
    """
    Uses the calculator above to generate new timestamps for Words and Phonemes.
    Skips writing a JSON sidecar file because Words and Phonemes don't need one.
    """
    if not starts:
        return
    split_starts, split_ends, split_labels = clip_vad_to_segments(
        vad_starts=starts, vad_ends=ends, vad_labels=labels, snd_segs=snd_segs,
    )
    lab_path = Path(str(stem_path) + ext)
    lab.store(str(lab_path), split_starts, split_ends, split_labels)


def remove_trailing_segments(base_dir: Path, stem_prefix: str, last_index: int) -> int:
    """
    Garbage collection. 
    If you ran this script yesterday and it made 10 chunks, but today you run it 
    with different settings and it only makes 8 chunks, this deletes chunks 9 and 10 
    so old data doesn't pollute your folders.
    """
    pattern = re.compile(rf"^{re.escape(stem_prefix)}_(\d+)(?:\.|$)")
    removed = 0
    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        # If the file number is higher than our total chunk count, delete it
        if int(match.group(1)) > last_index:
            path.unlink()
            removed += 1
    return removed


# ==============================================================================
# MAIN SCRIPT EXECUTION STARTS HERE
# ==============================================================================
if __name__ == '__main__':
    # 1. Set up the command-line arguments the script accepts
    parser = db_parser.create_db_iteration_parser(
        description="Split audio files based on VAD and align with texts",
        db_config_default=Path(__file__).parent.parent / "db_config.py",
        tags=["wav", "vad_loud"] # Look for these file types in the DB
    )
    parser.add_argument("-msd", "--max_silence_dur_s", default=None, type=float, required=True)
    parser.add_argument("-rsd", "--red_silence_dur_s", default=None, type=float)
    parser.add_argument("-mos", "--max_out_segment_dur_s", default=None, type=_duration_gt1, required=True)
    parser.add_argument("--trans_len_s", default=0., type=float)
    parser.add_argument("--snd_out_format", default=None)
    parser.add_argument("-mvd", "--min_voiced_dur_s", default=0.03, type=float)
    parser.add_argument("--output_dir", default="split_files", type=Path)
    parser.add_argument("-md", "--min_segment_dur", default=None, type=float)

    args = parser.parse_args()
    print(f"Database root: {args.root}")
    
    speaker_base: str | None = None
    base_dir_path: Path | None = None
    
    # 2. Ask the Agamemnon database for a list of all audio recordings to process
    for dd in dbn.generate_agamemnon_db_entries_from_tags(args.root, tags=args.tags,
                                                          beg_ind=args.begin_ind,
                                                          end_ind=args.end_ind,
                                                          dir_filter_pat=args.filter_pat,
                                                          error_on_missing=not args.no_check,
                                                          return_structure=True,
                                                          max_search_fac=args.max_search_fac):
        
        # 'base' tells us we are starting a new folder
        if "base" in dd:
            print(f"Processing base entry: {dd['base']}")
            base_dir_path = Path(cast(str, dd['base']))
            with (base_dir_path / 'meta.yml').open('r') as f:
                meta = yaml.safe_load(f)
            speaker_base = meta['speaker']
        
        # Otherwise, we are processing an actual audio file
        else:
            if speaker_base is None or base_dir_path is None:
                raise ValueError("No base entry found before data entries.")

            # Load the main Voice Activity Detection file (strictly validated)
            vad_starts, vad_ends, vad_labels = load_validated_vad_lab(cast(str, dd['vad_loud']))
            vad_path = Path(dd['vad_loud'])

            # Load the Phonemes dynamically (bypassing strict checks)
            phn_starts, phn_ends, phn_labels = [], [], []
            phn_path = Path(dd.get('phon', str(vad_path.parent.parent / "phon" / vad_path.name.replace(".vad.lab", ".phones.masked.lab"))))
            if phn_path.exists():
                phn_starts, phn_ends, phn_labels = load_generic_lab(phn_path)

            # Load the Words dynamically (bypassing strict checks)
            wrd_starts, wrd_ends, wrd_labels = [], [], []
            wrd_path = Path(dd.get('word', str(vad_path.parent.parent / "words" / vad_path.name.replace(".vad.lab", ".words.lab"))))
            if wrd_path.exists():
                wrd_starts, wrd_ends, wrd_labels = load_generic_lab(wrd_path)
            
            # Ensure root_path is a directory (handles if args.root is db_config.py)
            root_path = Path(args.root)
            if root_path.is_file():
                root_path = root_path.parent
            
            # Extract the exact folder structure relative to the root directory
            relative_dir = base_dir_path.relative_to(root_path)
            
            # Replicate this structure in the output directory
            cut_output_dir = args.output_dir / relative_dir
            
            # 3. PHYSICALLY SLICE THE AUDIO
            # This calls the external `cut_silences` tool to chop the `.wav` file up.
            ret = cut_silences(
                snd_file=cast(str, dd['wav']),
                vad_file=cast(str, dd['vad_loud']),
                output_dir=cut_output_dir,
                max_silence_dur_s=args.max_silence_dur_s,
                red_silence_dur_s=args.red_silence_dur_s,
                max_out_segment_dur_s=args.max_out_segment_dur_s, 
                trans_len_s=args.trans_len_s,
                snd_out_format=args.snd_out_format,
                min_voiced_dur_s=args.min_voiced_dur_s,
                return_silence_info=True,
                cmdline=f"split_vowels_with_texts.py snd={dd['wav']} vad={dd['vad_loud']}"
            )
            
            # Prepare to carry metadata over to the new files
            source_vad_json_path = Path(cast(str, dd['vad_loud']) + ".json")
            source_vad_meta = (
                json.loads(source_vad_json_path.read_text(encoding="utf-8"))
                if source_vad_json_path.exists()
                else {}
            )
            snd_basename = Path(cast(str, dd['wav'])).stem

            # ==================================================================
            # SONG MODE (M4Singer, OpenSinger, etc.)
            # ==================================================================
            # Loop through every new audio chunk created for this song
            for ind, seg_desc in enumerate(ret, start=1):
                # Figure out the new base name 
                split_base = cut_output_dir / f"{snd_basename}_{ind:04d}"
                
                # Safely handle the dictionary key depending on if the audio was grouped or not
                segments_list = seg_desc.get('snd_segs') or [seg_desc['snd_seg']]
                
                # 1. Slice and save the VAD file to match this audio chunk
                write_split_vad(
                    stem_path=split_base,
                    snd_segs=segments_list,
                    vad_starts=vad_starts,
                    vad_ends=vad_ends,
                    vad_labels=vad_labels,
                    source_vad_meta=source_vad_meta,
                    source_vad_path=cast(str, dd['vad_loud']),
                )
                
                # 2. Slice and save the Phonemes (if they were found earlier)
                write_split_generic_lab(
                    stem_path=split_base,
                    snd_segs=segments_list,
                    starts=phn_starts,
                    ends=phn_ends,
                    labels=phn_labels,
                    ext=".phones.masked.lab"
                )
                
                # 3. Slice and save the Words (if they were found earlier)
                write_split_generic_lab(
                    stem_path=split_base,
                    snd_segs=segments_list,
                    starts=wrd_starts,
                    ends=wrd_ends,
                    labels=wrd_labels,
                    ext=".words.lab"
                )
                
            # Garbage collection: delete leftover old chunks if any
            removed = remove_trailing_segments(cut_output_dir, snd_basename, len(ret))
            if removed:
                print(f"Removed {removed} trailing stale split files for {speaker_base}/{snd_basename}")