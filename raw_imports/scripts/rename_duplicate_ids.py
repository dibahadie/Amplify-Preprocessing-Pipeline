import argparse
from pathlib import Path
from collections import defaultdict

def rename_duplicate_ids(root_dir, dry_run=False):
    root = Path(root_dir)
    
    # Structure: song_paths[song_id] = {'F': [path1, ...], 'M': [path2, ...]}
    song_paths = defaultdict(lambda: {'F': [], 'M': []})
    
    # 1. Map all Song_IDs and their paths
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir(): continue
        for gender_dir in lang_dir.iterdir():
            gender = gender_dir.name
            if gender not in ['F', 'M']: continue
            
            for song_dir in gender_dir.iterdir():
                if not song_dir.is_dir(): continue
                song_id = song_dir.name
                song_paths[song_id][gender].append(song_dir)

    # 2. Find overlaps and rename
    renamed_count = 0
    for song_id, paths_by_gender in song_paths.items():
        if paths_by_gender['F'] and paths_by_gender['M']:
            # The Song_ID exists in both F and M
            
            # Process Male paths (append '00')
            for m_path in paths_by_gender['M']:
                new_m_path = m_path.parent / f"{song_id}00"
                if dry_run:
                    print(f"[DRY RUN - RENAME MALE] {m_path} -> {new_m_path.name}")
                else:
                    m_path.rename(new_m_path)
                    
            # Process Female paths (append '11')
            for f_path in paths_by_gender['F']:
                new_f_path = f_path.parent / f"{song_id}11"
                if dry_run:
                    print(f"[DRY RUN - RENAME FEMALE] {f_path} -> {new_f_path.name}")
                else:
                    f_path.rename(new_f_path)
            
            renamed_count += 1

    if dry_run:
        print(f"\n[DRY RUN] Found {renamed_count} overlapping Song_IDs to rename.")
    else:
        print(f"\nSuccessfully renamed {renamed_count} overlapping Song_IDs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differentiate overlapping Song_IDs between F and M.")
    parser.add_argument("--root", required=True, help="Path to the main dataset directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without modifying the disk.")
    
    args = parser.parse_args()
    
    rename_duplicate_ids(args.root, args.dry_run)