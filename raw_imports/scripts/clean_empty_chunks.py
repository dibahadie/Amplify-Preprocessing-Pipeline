import os
import argparse
from pathlib import Path
from collections import defaultdict

def clean_empty_chunks(root_dir, dry_run=False):
    root = Path(root_dir)
    
    # Traverse Language / Gender / Song_ID directories
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir(): continue
        for gender_dir in lang_dir.iterdir():
            if not gender_dir.is_dir(): continue
            for song_dir in gender_dir.iterdir():
                if not song_dir.is_dir(): continue
                
                auto_txt_dir = song_dir / "auto_txt"
                if not auto_txt_dir.exists():
                    continue
                    
                # Group chunks by their 'n' prefix so we sequence 'm' correctly per song
                # Structure: n_groups[n_prefix] = [(m_int, m_str, txt_file_path), ...]
                n_groups = defaultdict(list)
                
                for txt_file in auto_txt_dir.glob("*_*.txt"):
                    base_name = txt_file.stem  # e.g., "0_0001"
                    try:
                        n_prefix, m_str = base_name.split('_')
                        n_groups[n_prefix].append((int(m_str), m_str, txt_file))
                    except ValueError:
                        continue
                        
                for n_prefix, chunks in n_groups.items():
                    # Sort by the integer value of 'm' to process in original sequential order
                    chunks.sort(key=lambda x: x[0])
                    
                    new_m_counter = 1
                    
                    for m_int, m_str, txt_path in chunks:
                        old_base = f"{n_prefix}_{m_str}"
                        
                        # Check if the text file is empty (0 bytes or only whitespace)
                        is_empty = False
                        if txt_path.stat().st_size == 0:
                            is_empty = True
                        else:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                if not f.read().strip():
                                    is_empty = True
                        
                        # Gather all associated files for this chunk across all directories
                        related_files = []
                        # Files in the main song_dir (.flac, .json, .vad.lab, etc.)
                        for f in song_dir.glob(f"{old_base}.*"):
                            if f.is_file(): related_files.append(f)
                        # The auto_txt file itself
                        if txt_path.exists(): related_files.append(txt_path)
                        # The auto_word file
                        auto_word_file = song_dir / "auto_word" / f"{old_base}.word.lab"
                        if auto_word_file.exists(): related_files.append(auto_word_file)
                        
                        if is_empty:
                            # Delete the chunk completely
                            for file_to_delete in related_files:
                                if dry_run:
                                    print(f"[DRY RUN - DELETE] {file_to_delete}")
                                else:
                                    file_to_delete.unlink()
                        else:
                            # Keep it, but check if we need to rename it to close gaps
                            new_m_str = f"{new_m_counter:04d}"
                            new_base = f"{n_prefix}_{new_m_str}"
                            
                            if old_base != new_base:
                                for file_to_rename in related_files:
                                    parent = file_to_rename.parent
                                    # Capture the exact extension (.flac, .vad.lab.json, etc.)
                                    extension = file_to_rename.name[len(old_base):] 
                                    new_path = parent / f"{new_base}{extension}"
                                    
                                    if dry_run:
                                        print(f"[DRY RUN - RENAME] {file_to_rename} -> {new_path.name}")
                                    else:
                                        file_to_rename.rename(new_path)
                                        
                            new_m_counter += 1

    print(f"\nCleanup complete. {'(DRY RUN - no files modified)' if dry_run else ''}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove empty chunks and reorganize sequence numbers.")
    parser.add_argument("--root", required=True, help="Path to the main reorganized dataset directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without modifying the disk.")
    
    args = parser.parse_args()
    
    clean_empty_chunks(args.root, args.dry_run) 