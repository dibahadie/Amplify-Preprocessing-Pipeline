#! /usr/bin/env bash
#DOC Usage: scripts/transcribe_split_songs.sh [options]
#DOC
#DOC Transcribe split song audio files with whisper-ctranslate2.
#DOC Language is automatically detected by Whisper for each file.
#DOC
#DOC Options:
#DOC   -o, --output_dir DIR       Directory where transcription folders are written
#DOC                              (default: RAW_IMPORTS_DIR/split_files)
#DOC   -i, --input PATH           Input root directory containing Language/Gender/SongDir structure
#DOC                              (default: RAW_IMPORTS_DIR/Amplify)
#DOC   -a, --audio_ext EXT        Audio extension used for directory scans
#DOC                              (default: .flac)
#DOC   -m, --model NAME           whisper-ctranslate2 model (default: large-v3)
#DOC   -f, --output_format FMT    whisper output format (default: txt)
#DOC   -c, --compute TYPE         ctranslate2 compute type. Defaults to float16
#DOC   --force                    Overwrite existing transcriptions (default: skip if exists)
#DOC   --override_dir DIR         Directory with guarded transcript overrides
#DOC   --override_hashes FILE     Audio hash JSON used to validate overrides
#DOC   --no_overrides             Disable transcript overrides.
#DOC   --conda_env NAME           Conda environment to activate (default: VOMO)
#DOC   --no_conda                 Do not activate a conda environment
#DOC   -n, --dry_run              Print the command without running it
#DOC   -h, --help                 Show this help

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_IMPORTS_DIR="$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_SPLIT_DIR="${RAW_IMPORTS_DIR}/split_files"
MODEL="large-v3"
OUTPUT_FORMAT="txt"
OUTPUT_DIR="${DEFAULT_SPLIT_DIR}/Amplify"
AUDIO_EXT=".flac"
INPUT_PATH="${RAW_IMPORTS_DIR}/split_files/Amplify"

OVERRIDE_DIR="${RAW_IMPORTS_DIR}/split_files/txt_overrides"
OVERRIDE_HASHES="${RAW_IMPORTS_DIR}/split_files/songs_split_hashes.json"
OVERRIDES_ENABLED=1
OVERRIDE_DIR_FROM_CLI=0
CONDA_ENV="tf2.14"
USE_CONDA=1
DRY_RUN=0
COMPUTE=""
FORCE_TRANSCRIBE=0

usage() {
    sed -n 's/^#DOC \{0,1\}//p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -i|--input) INPUT_PATH="$2"; shift 2 ;;
        -a|--audio_ext) AUDIO_EXT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        -f|--output_format) OUTPUT_FORMAT="$2"; shift 2 ;;
        -c|--compute) COMPUTE="$2"; shift 2 ;;
        --force) FORCE_TRANSCRIBE=1; shift ;;
        --override_dir) OVERRIDE_DIR="$2"; OVERRIDE_DIR_FROM_CLI=1; shift 2 ;;
        --override_hashes) OVERRIDE_HASHES="$2"; shift 2 ;;
        --no_overrides) OVERRIDES_ENABLED=0; shift ;;
        --conda_env) CONDA_ENV="$2"; shift 2 ;;
        --no_conda) USE_CONDA=0; shift ;;
        -n|--dry_run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${COMPUTE}" ]]; then
    COMPUTE="int8"
    if [[ "$(uname)" == "Darwin" ]]; then COMPUTE="float32"; fi
fi

if [[ "${AUDIO_EXT}" != .* ]]; then AUDIO_EXT=".${AUDIO_EXT}"; fi

if [[ ! -d "${INPUT_PATH}" ]]; then
    echo "Input path does not exist or is not a directory: ${INPUT_PATH}" >&2
    exit 1
fi

APPLY_OVERRIDES=0
if [[ ${OVERRIDES_ENABLED} -eq 1 ]]; then
    if [[ -d "${OVERRIDE_DIR}" ]]; then
        APPLY_OVERRIDES=1
        if [[ "${OUTPUT_FORMAT}" != "txt" ]]; then
            echo "Transcript overrides require --output_format txt." >&2; exit 2
        fi
        if [[ ${DRY_RUN} -eq 0 && ! -f "${OVERRIDE_HASHES}" ]]; then
            echo "Transcript override hashes not found: ${OVERRIDE_HASHES}" >&2; exit 1
        fi
    elif [[ ${OVERRIDE_DIR_FROM_CLI} -eq 1 ]]; then
        echo "Transcript override directory not found: ${OVERRIDE_DIR}" >&2; exit 1
    fi
fi

if [[ ${USE_CONDA} -eq 1 ]]; then
    : "${ANACONDA:?ANACONDA must point to the conda installation, or use --no_conda}"
    source "${ANACONDA}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# ==============================================================================
# Set Up Logging
# ==============================================================================
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/whisper_log.txt"

if [[ ${DRY_RUN} -eq 0 ]]; then
    echo "=========================================================" >> "${LOG_FILE}"
    echo "Starting new batch transcription run at $(date)" >> "${LOG_FILE}"
    echo "=========================================================" >> "${LOG_FILE}"
fi

# ==============================================================================
# Main Processing Loop
# ==============================================================================

# Search 3 levels deep: InputPath/Language/Gender/SongDir/
shopt -s nullglob
SONG_DIRS=("${INPUT_PATH}"/*/*/*/)
shopt -u nullglob

if [[ ${#SONG_DIRS[@]} -eq 0 ]]; then
    echo "No subdirectories found 3 levels deep in ${INPUT_PATH}." >&2
    echo "Expected structure: ${INPUT_PATH}/<Language>/<Gender>/<SongDir>/" >&2
    exit 1
fi

for SONG_DIR in "${SONG_DIRS[@]}"; do
    # Remove trailing slash for safe path parsing
    CLEAN_PATH="${SONG_DIR%/}"
    
    # Extract the directory names by working backwards up the path
    SONG_ID=$(basename "${CLEAN_PATH}")
    GENDER=$(basename "$(dirname "${CLEAN_PATH}")")
    LANGUAGE=$(basename "$(dirname "$(dirname "${CLEAN_PATH}")")")
    
    shopt -s nullglob
    SONG_FILES=("${SONG_DIR}"*"${AUDIO_EXT}")
    shopt -u nullglob
    
    if [[ ${#SONG_FILES[@]} -eq 0 ]]; then
        echo "Skipping ${SONG_ID}: No ${AUDIO_EXT} files found." | tee -a "${LOG_FILE}"
        continue
    fi

    # Target directory mimics input structure: Output/Language/Gender/SongDir/
    CURRENT_OUT_DIR="${OUTPUT_DIR}/${LANGUAGE}/${GENDER}/${SONG_ID}/auto_txt"
    
    # Filter files that already have transcriptions
    FILES_TO_PROCESS=()
    for audio_file in "${SONG_FILES[@]}"; do
        base_name=$(basename "${audio_file}")
        expected_out="${CURRENT_OUT_DIR}/${base_name%.*}.${OUTPUT_FORMAT}"
        
        # Check if output file exists (-f)
        if [[ ${FORCE_TRANSCRIBE} -eq 0 && -f "${expected_out}" ]]; then
            echo "Skipping ${base_name}: Transcription already exists." | tee -a "${LOG_FILE}"
        else
            FILES_TO_PROCESS+=("${audio_file}")
        fi
    done

    if [[ ${#FILES_TO_PROCESS[@]} -eq 0 ]]; then
        echo "All files in ${SONG_ID} are already transcribed. Skipping folder." | tee -a "${LOG_FILE}"
        continue
    fi
    
    echo "=========================================================" | tee -a "${LOG_FILE}"
    echo "Processing Song ID: ${SONG_ID} (${LANGUAGE} / ${GENDER})" | tee -a "${LOG_FILE}"
    echo "Transcribing ${#FILES_TO_PROCESS[@]} out of ${#SONG_FILES[@]} files" | tee -a "${LOG_FILE}"
    echo "=========================================================" | tee -a "${LOG_FILE}"
    
    cmd=(
        whisper-ctranslate2
        --model="${MODEL}"
        "${FILES_TO_PROCESS[@]}"
        --compute "${COMPUTE}"
        --output_format "${OUTPUT_FORMAT}"
        --output_dir "${CURRENT_OUT_DIR}"
    )
    
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'mkdir -p %q\n' "${CURRENT_OUT_DIR}"
        printf '%q ' "${cmd[@]}"
        printf ' 2>&1 | tee -a %q\n\n' "${LOG_FILE}"
        if [[ ${APPLY_OVERRIDES} -eq 1 ]]; then
            printf 'python3 %q --override_dir %q --hashes %q --output_dir %q --dry_run ' \
                "${SCRIPT_DIR}/apply_transcription_overrides.py" "${OVERRIDE_DIR}" "${OVERRIDE_HASHES}" "${CURRENT_OUT_DIR}"
            printf '%q ' "${FILES_TO_PROCESS[@]}"
            printf ' >> %q 2>&1\n\n' "${LOG_FILE}"
        fi
    else
        mkdir -p "${CURRENT_OUT_DIR}"
        # Run Whisper and pipe both stdout and stderr to the log file and screen
        "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
        
        if [[ ${APPLY_OVERRIDES} -eq 1 ]]; then
            python3 "${SCRIPT_DIR}/apply_transcription_overrides.py" \
                --override_dir "${OVERRIDE_DIR}" \
                --hashes "${OVERRIDE_HASHES}" \
                --output_dir "${CURRENT_OUT_DIR}" \
                "${FILES_TO_PROCESS[@]}" 2>&1 | tee -a "${LOG_FILE}"
        fi
    fi
done

echo "Batch processing complete. Logs saved to ${LOG_FILE}" | tee -a "${LOG_FILE}"