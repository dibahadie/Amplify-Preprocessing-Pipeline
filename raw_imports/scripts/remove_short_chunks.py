import os
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

def get_audio_duration(file_path):
    """Returns the duration of the audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        # Fallback to soxi if ffprobe fails or is missing
        try:
            result = subprocess.run(
                ['soxi', '-D', str(file_path)],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            print(f"Error reading duration for {file_path}. Skipping.")
            return None

def remove_and_reorganize_short_chunks(root_dir, dry_run=False):
    root = Path(root_dir)
    
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir(): continue
        for gender_dir in lang_dir.iterdir():
            if not gender_dir.is_dir(): continue
            for song_dir in gender_dir.iterdir():
                if not song_dir.is_dir(): continue
                
                # Group chunks by their 'n' prefix
                # Structure: n_groups[n_prefix] = [(m_int, m_str, flac_path), ...]
                n_groups = defaultdict(list)
                
                for flac_file in song_dir.glob("*_*.flac"):
                    base_name = flac_file.stem
                    try:
                        n_prefix, m_str = base_name.split('_')
                        n_groups[n_prefix].append((int(m_str), m_str, flac_file))
                    except ValueError:
                        continue
                
                for n_prefix, chunks in n_groups.items():
                    # Sort sequentially by original 'm' value
                    chunks.sort(key=lambda x: x[0])
                    
                    new_m_counter = 1
                    
                    for m_int, m_str, flac_path in chunks:
                        old_base = f"{n_prefix}_{m_str}"
                        
                        duration = get_audio_duration(flac_path)
                        if duration is None:
                            continue
                        
                        is_short = duration < 1.0
                        
                        # Gather all related files across subdirectories
                        related_files = []
                        for f in song_dir.glob(f"{old_base}.*"):
                            if f.is_file(): related_files.append(f)
                            
                        auto_txt_file = song_dir / "auto_txt" / f"{old_base}.txt"
                        if auto_txt_file.exists(): related_files.append(auto_txt_file)
                        
                        auto_word_file = song_dir / "auto_word" / f"{old_base}.word.lab"
                        if auto_word_file.exists(): related_files.append(auto_word_file)
                        
                        if is_short:
                            # Delete the short chunk and all associated files
                            for file_to_delete in related_files:
                                if dry_run:
                                    print(f"[DRY RUN - DELETE] {file_to_delete} (Duration: {duration:.2f}s)")
                                else:
                                    file_to_delete.unlink()
                        else:
                            # Keep it, and rename if sequence gap exists
                            new_m_str = f"{new_m_counter:04d}"
                            new_base = f"{n_prefix}_{new_m_str}"
                            
                            if old_base != new_base:
                                for file_to_rename in related_files:
                                    parent = file_to_rename.parent
                                    extension = file_to_rename.name[len(old_base):]
                                    new_path = parent / f"{new_base}{extension}"
                                    
                                    if dry_run:
                                        print(f"[DRY RUN - RENAME] {file_to_rename} -> {new_path.name}")
                                    else:
                                        file_to_rename.rename(new_path)
                                        
                            new_m_counter += 1

    print(f"\nCleanup complete. {'(DRY RUN - no files modified)' if dry_run else ''}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove audio chunks < 1s and reorganize sequence numbers.")
    parser.add_argument("--root", required=True, help="Path to the main reorganized dataset directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without modifying the disk.")
    
    args = parser.parse_args()
    
    remove_and_reorganize_short_chunks(args.root, args.dry_run)