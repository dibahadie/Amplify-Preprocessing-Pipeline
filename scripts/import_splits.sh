#! /usr/bin/env bash

DIR=$(cd $(dirname "${BASH_SOURCE[0]}") > /dev/null && pwd)
export AGAMEMNON_IMPORT_ROOT="$DIR/.."
cd "$AGAMEMNON_IMPORT_ROOT"

${AS_PYSRC}/fileio/speechdb/agamemnon_import_data.py ./db_config.py --force
# The follow-up import copies split text content for songs and vowels, phoneme
# sequences for songs, aligned vowel phone labels, and split/clipped VAD results
# into the target speechdb layout. Song lab_phon files remain virtual labels with
# invalid times (-1), to be replaced later by aligned phone timings if needed.
./scripts/import_txt_and_labs.py

