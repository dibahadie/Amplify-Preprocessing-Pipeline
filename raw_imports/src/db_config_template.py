import os
from collections import OrderedDict
from fileio.speechdb.db_names import db_config_spec_interface
import json
import re
from pathlib import Path
from functools import lru_cache

class db_config_spec(db_config_spec_interface):
    """
    Example class for speechdb directory specification
    """

    def __init__(self, dbroot):
        """
        db config class providing a directory structure for speech corpora
        should be placed in a db_config.py file of a speech database.

        The the constructor for the basic class providing directory information about all files accessible
        in a voice database supported by fileio.speechdb. Can also serve as an end point directory in
        an Agamemnon multi speaker database.

        Individual files can be retrieved by means of the `get` function that accepts an unused
        first argument followed by the desired tag and the file index.

        fileio.speechdb provides mutlple functions for accessing and iterating files in a database,
        and scripts for listing, copying, and running analysis or synthesis algorithms with such a database.

        To support the get function a db dictionary is used that maps database tags
        to strings with a single format entry receiving the index. Mapping is achieved by means of a dictionary
        mapping tag-strings to format string that has a single format specifier that can format an int.

        Assume the speech database has directory layout as follows:

          |
          |  dbroot/audio/wav/1.wav
          |  dbroot/txt/1.txt
          |  dbroot/seq/phn/1.mrk.sdif
          |  dbroot/analysis/f0/1.f0.sdif
          |

        The the db dict creating the mapping between tags and filenames could become

        **Example db argument:**
           |
           |   {"wav": os.path.join(dbroot, "audio", "wav","{0:d}.wav"),
           |   "txt": os.path.join(.dbroot, "txt", "{0:d}.txt"),
           |   "mrk": os.path.join(dbroot, "seq", "phn", "{0:d}.mrk.sdif"),
           |   "f0": os.path.join(dbroot, "analysis", "f0", "{0:d}.f0.sdif"),
           |   }

        **Parameters:**
        :param dbroot: absolut path to db_config file containing this class
        :type dbroot: str
        :param db: dict containing a strings with a single format spec receiving an int
           that indicates the index of the file for that the name should be produced.
        :type db: dict[str, str]
        """
        self.db_root = dbroot
        self.db = {"wav": os.path.join(self.db_root, "audio", "wav","{0:d}.flac"),
                   "txt": os.path.join(self.db_root, "txt", "{0:d}.txt"),
                   "vad_loud": os.path.join(self.db_root, "seq", "vad_loud", "{0:d}.vad.lab"),
                   "phon": os.path.join(self.db_root, "seq", "phon", "{0:d}.phn"),
                   "phon_fl": os.path.join(self.db_root, "seq", "phon_fl", "{0:d}.phn"),
                   "phon_esp": os.path.join(self.db_root, "seq", "phon_esp", "{0:d}.phn"),
                   "phon_lia": os.path.join(self.db_root, "seq", "phon_lia", "{0:d}.phn"),
                   "f0_FCN": os.path.join(self.db_root, "analysis", "f0", "{0:d}.f0.sdif"),
                   "f0h_r": os.path.join(self.db_root, "analysis", "f0h_r", "{0:d}.f0h_r.sdif"),
                   "f0h_z": os.path.join(self.db_root, "analysis", "f0h_z", "{0:d}.f0h_z.sdif"),
                   }

    def get(self, base, typ, num):
        # find proper db entry
        return self.db[typ].format(num)

    def root(self):
        return self.db_root

@lru_cache(maxsize=1)
def load_language_map() -> dict:
    """
    Parses sources.yml and lyrics.json exactly like the bash script to map 
    song_ids (e.g., '10836') to their languages (e.g., 'en', 'fr').
    """
    lang_map = {}
    if not SOURCES_YML.exists() or not LYRICS_JSON.exists():
        return lang_map

    # 1. Load lyrics mapping: original_string -> language
    original_str_to_lang = {}
    try:
        with open(LYRICS_JSON, "r", encoding="utf-8") as f:
            for song in json.load(f):
                if "original_string" in song and "language" in song:
                    original_str_to_lang[song["original_string"]] = song["language"]
    except Exception as e:
        print(f"Warning: Failed to parse lyrics.json: {e}")

    # 2. Map song_ids to original_strings via sources.yml
    try:
        with open(SOURCES_YML, "r", encoding="utf-8") as f:
            for line in f:
                if "ORI:" in line:
                    # Look for the pattern: /original_string_12345/
                    match = re.search(r"/([^/]+)_(\d+)(?:/|_)", line)
                    if match:
                        original_string = match.group(1)
                        numeric_id = match.group(2)
                        if original_string in original_str_to_lang:
                            lang_map[numeric_id] = original_str_to_lang[original_string]
    except Exception as e:
        print(f"Warning: Failed to parse sources.yml: {e}")

    return lang_map


source_file = __file__
path = os.path.dirname(source_file)
template_file = os.path.realpath(__file__)
template_path = os.path.dirname(template_file)
french_word_override_file = os.path.join(template_path, "french_word_overrides.txt")
db_config = db_config_spec(path)
SOURCES_YML = Path(os.path.join(template_path, "../Tency/sources.yml"))
LYRICS_JSON = Path(os.path.join(template_path, "../lyrics.json"))
english_word_override_file = os.path.join(template_path, "english_word_overrides.txt")

###############################################################
# Analysis parameter section
# see documentation in fileio.speechdb.create_analysis for
# the supported analysis algorithms.
###############################################################
f0min = 50
f0max = 800
f0max_for_win = 200

params = OrderedDict()

params['f0_FCN'] = {
    "step_s": 0.002,
    "min_f0_limit" : 100,
    "max_f0_limit" : 1100,
}

params['f0h_z']  ={ "f0file"                :  "=f0_FCN",
                    "remove_low_harm_f0"    : False,
                    "harmonicity_threshold" : 0.5,
                    "confidence_threshold" : 0.5,
                    "unvoiced_min_length_s" : 0.020,
                    "voiced_min_length_s" : 0.020,
                }

params['f0h_r']  ={ "f0file"                :  "=f0_FCN",
                    "remove_low_harm_f0"    : True,
                    "harmonicity_threshold" : 0.5,
                    "confidence_threshold" : 0.5,
                    "unvoiced_min_length_s" : 0.020,
                    "voiced_min_length_s" : 0.020,
                }

params['phon'] = {
    # this will be calling festival
    "infile": "=txt",
    "language": "en-us",
}

params['phon_fl'] = {
    "infile": "=txt",
    "language": "en-us",
    "backend" : "flite",
    "split_diphtongs": True,
    "force_silence_start": True,
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "debug": False,
}

params['phon_esp'] = {
    "infile": "=txt",
    "language": "en-us",
    "backend" : "espeak",
    "force_silence_start": False,
    "add_word_separators": True,
    "separate_words": True,
    "split_diphtongs": True,
    "normalized_text_suffix": ".ntxt",
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "remove_punctuation": True,
    "text_lower_case": True,
    "debug": False,
    "override_file": english_word_override_file,
}

params['phon_esp_fr'] = {
    "infile": "=txt",
    "language": "fr-fr",
    "backend" : "espeak",
    "force_silence_start": False,
    "add_word_separators": True,
    "separate_words": True,
    "split_diphtongs": True,
    "normalized_text_suffix": ".ntxt",
    "override_file": french_word_override_file,
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "remove_punctuation": True,
    "text_lower_case": True,
    "espeak_lower_case_nasals": True,
    "debug": False,
}

params['phon_lia'] = {
    "infile": "=txt",
    "backend" : "liaphon",
    "force_silence_start": False,
    "add_word_separators": True,
    "separate_words": True,
    "split_diphtongs": True,
    "normalized_text_suffix": ".ntxt",
    "override_file": french_word_override_file,
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "remove_punctuation": True,
    "text_lower_case": True,
    "debug": False,
}

params['phon_lia_vax'] = {
    "infile": "=txt",
    "backend" : "liaphon",
    "force_silence_start": True,
    "add_word_separators": True,
    "separate_words": False,
    "split_diphtongs": True,
    "normalized_text_suffix": ".n_vax_txt",
    "override_file": french_word_override_file,
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "remove_punctuation": True,
    "text_lower_case": True,
    "liaphon_variant_mode": "INSERTIONS_MERGED",
    "debug": False,
}



params['split_phon_esp'] = {
    "infile": "=txt_split",
    "language": "en-us",
    "backend" : "espeak",
    "force_silence_start": True,
    "add_word_separators": True,
    "split_diphtongs": True,
    "convert_xsampa" : True,
    "use_lab_format" : False,
    "debug": False,
}


params["align_phon"] = {
    "infile"  : "=wav",
    "txtfile" : "=txt",
    "language" : "en",
    "mix_noise_level_snr_db" : 40,
    }

params["vax_words"] = {
    "infile"  : "=wav",
    "text_filepath" : "=txt",
    "model_tag" : "s6",
    "save_format": "audacity",
    "save_delimiter":"\t",
    }

params['vad_loud'] = {
   "speech_duration_min_s" : 0.100 if "vowel" in path else 0.020,
   "silence_duration_min_s" : 0.020 if "vowel" in path else 0.200,
   "params" : {
      "mode": "loud",
      "loud_alpha" : 0.30103,
      "loud_meanVoiceLevelDb": 60.,
      "loud_sig_gain_factor": 2.,
      "loud_silence_thresh_sone": None,
      "silence_thresh_auto_std": 0.95,
      "loud_silence_sone_modulation_fac": 2.,
      "no_hearing_threshold": False,
      "min_loudness": None
   }
}


if __name__ == '__main__':
    print("dbconfig has keys")
    for key_ in db_config.db:
        print(key_)
    print("\n\n")
    print("Param dict has the keys")
    for key_ in params:
        print(key_)
    print("\nDETAILS:")
    for key_, value_ in params.items():
        print(f"{key_}:")
        for k, v in value_.items():
            print(f"  {k}: {v}")
