# 1. Total .flac files actually in --root (follows symlinks) vs lines in your list
find -L split_files/Amplify -type f -name "*.flac" | wc -l
wc -l scripts/dataset_structure.txt

# 2. Everything matching 15077 on disk under --root (follows symlinks)
find -L split_files/Amplify -path "*37670*"

# 3. Everything matching 15077 in your target list
grep "37670" scripts/dataset_structure.txt

# French/M/14433/0_0011.flac
# Amplify-reproduce/raw_imports/scripts/dataset_structure.txt
# English/F/37670