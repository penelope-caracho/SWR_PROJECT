from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent

# config.yaml laden
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

CFG = _raw_cfg  

# Pfade
SENTIWS_PATH = BASE_DIR / CFG["paths"]["sentiws"]
FASTTEXT_PATH = BASE_DIR / CFG["paths"]["fasttext"]
MODEL_DIR = BASE_DIR / CFG["paths"]["model_dir"]

# Server / API
SERVER_HOST = CFG["server"]["host"]
SERVER_PORT = CFG["server"]["port"]
API_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/generate"

# Phonemizer / espeak-ng
ESPEAK_LIB_PATH = CFG["phonemizer"]["espeak_library"]

# Evaluation (Plausibilitätsprüfung)
SIMILARITY_THRESHOLD = CFG["evaluation"]["similarity_threshold"]
FREQ_MIN = CFG["evaluation"]["freq_min"]
FREQ_MIN_OOV_ADJ = CFG["evaluation"]["freq_min_oov_adj"]
FREQ_MIN_OOV_NOUN = CFG["evaluation"]["freq_min_oov_noun"]
FREQ_HARD_MIN = CFG["evaluation"]["freq_hard_min"]

# Sentiment
SENTIMENT_NEG_THRESHOLD = CFG["sentiment"]["neg_threshold"]

# Generator (GPT-2)
GEN_MAX_LENGTH = CFG["generator"]["max_length"]
GEN_NUM_RETURN_SEQUENCES = CFG["generator"]["num_return_sequences"]
GEN_TOP_K = CFG["generator"]["top_k"]
GEN_TOP_P = CFG["generator"]["top_p"]
GEN_TEMPERATURE = CFG["generator"]["temperature"]
