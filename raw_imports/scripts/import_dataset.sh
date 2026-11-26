#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DIR="$(cd "$SCRIPT_DIR/.." > /dev/null && pwd -P)"
cd "$DIR"

# import the audio and text
agamemnon_import_data.py . --force
run_analysis.py db_config.py --tags vad_loud
python scripts/split_vowels_with_texts.py -msd 0.2 --max_out_seg 20 -md 1
export HF_HOME=/data2/anasynth_nonbp/hadi/.cache/huggingface
scripts/transcribe_split_songs.sh

python_gpu_lock -- python scripts/remove_reverb.py

python clean_whisper_logs.py --root split_files/Amplify --log /path/to/transcription_log.txt
python generate_labels.py --root split_files/Amplify --log /path/to/transcription_log.txt
python clean_empty_chunks.py --root split_files/Amplify
python rename_duplicate_ids.py --root split_files/Amplify
python remove_short_chunks.py --root split_files/Amplify


