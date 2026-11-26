## Introduction

This project has been done in an internship by `Diba Hadi Esfangereh` under the supervision of `Axel Roebel`. If there is any questions regarding the code, feel free to contact the author at `diba.hadie@gmail.com`.

The following material contains the code for reproducing the `AudioWhisperMultiLang` version of the Amplify dataset from `MusicSeparationData_Flac`. This dataset is constructed for synthesis of singing voice and contains the singing stems provided in the Amplify dataset.

Several preprocessing steps have been done to clean up the data:

1. **Song Transcription**: the original dataset has not provided the trancriptions of the pieces. This has been done with `Whisper`.

2. **Gender and Language Organization**: Although the gender of the singer is given in the stem number (9220 for male and 9210 for female), the language of the songs has not been provided. Language detection has been done with the use of whisper logs. Later on, alignment models were trained on the cleaned dataset and the misclassified song languages were caught as the model will catch an error if the song and the transcription are completely different.

3. **Slicing and Silence Removal**: Each track is very long and contains many silent sections. This has been done using the scripts provided in the agamemnon database.

4. **Initial Filtering**: After initial slicing, several silent segments are introduced. These could be breath sounds or very long vowels not transcribed by whisper. A script goes over the dataset and removes all samples with empty transcriptions.

5. **Generating Word Lables**: As whisper trascribes the songs, it prints times segments of the transcription in the log. Although this is not very accurate, a script goes over the log and stores these.

6. **Reverb Removal**: The initial dataset contains a lot of reverb. This has been done with the use of the following work. `https://github.com/sp-uhh/sgmse`

7. **Handeling Songs with Multiple Singers**: Some songs have multiples singers. As the dataset is organized by song_id, this will create problems. A script goes over such songs, and seperates the singers by creating new song ids for each singer.

The current version is at `/net/assdb/data/music/Amplify/AudioWhisperMultiLang`.

You should do the preparations needed to support the `agamemnon` database before following with the rest of the instructions.


## Initial Import

All of the steps in this section must be done in the `raw_imports` folder.

```
cd raw_imports
```

For importing the audio files organized into gender folders, run the following command. This will go over the original dataset and import audio files grouped be their song id and stem id.

```
agamemnon_import_data.py . --force
```
Generate the voice activity detection data and split to segments less than 20 seconds.

```
run_analysis.py db_config.py --tags vad_loud
python scripts/split_vowels_with_texts.py -msd 0.2 --max_out_seg 20 -md 1
```

## Transcription

Transcribe the split segments. You may need to configure HF_HOME from your home directory which doesn't have enough space to somewhere else.

**Note**: In the current version of dataset, dereverberation was done after transcription as I discovered the audio quality issues later. For perfect reproduction of that version you can do the same. For future versions, running the dereverberation before transcription might be beneficial. 

```
export HF_HOME=/path/to/storage/with/space
scripts/transcribe_split_songs.sh
```

## Reorganization

To reorganize the dataset by language and generate the times word labels from a given whisper log, run the following. 

```
<!-- The following commands should be ran in the `split_files` folder. -->
python scripts/clean_whisper_logs.py --root Amplify --log /path/to/transcription_log.txt
python scripts/generate_labels.py --root Amplify --log /path/to/transcription_log.txt
```

If you wish to recreate the txt files using a provided whisper log, run the following. This creates the transcriptions from the .word.lab files.

```
python scripts/generate_auto_txt.py --root split_files/Amplify
```

Files with empty transcriptions or files that are too short are filtered.

```
python scripts/clean_empty_chunks.py --root split_files/Amplify
python scripts/remove_short_chunks.py --root split_files/Amplify
```

Some songs have multiple singers, which can be problematic since we are grouping by song id. For the safe handeling of this, we have created new song ids by appending 00 (male) and 11 (female) to the song id and creating seperate folders for them.

```
python rename_duplicate_ids.py --root Amplify
```

### Reverb Removal

Download the pretrained checkpoints and run using the following.

```
curl -L "https://www2.informatik.uni-hamburg.de/sp/audio/publications/itg2025-reverbfx/checkpoints/sgmse_artificial_rir_350k.ckpt" -o scripts/sgmse/sgmse_artificial_rir_350k_2.ckpt
source scripts/sgmse/env/bin/activate
pip install -e scripts/sgmse
python_gpu_lock -- python scripts/remove_reverb.py
```


