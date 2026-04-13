from pathlib import Path

# ---------------------------------------------------------------------------
# Ollama models to test (must be pulled locally: `ollama pull <model>`)
# ---------------------------------------------------------------------------
OLLAMA_MODELS = [
    "llama3.1:latest",
    "mistral:7b",
    "phi3.5:latest",
]

# ---------------------------------------------------------------------------
# Augmentation scenarios
# Each ratio defines both how much real data is REMOVED and how much
# synthetic data is ADDED back, so the total dataset size stays constant:
#   real = (1 - ratio) * N  |  synthetic = ratio * N  |  total = N
# ---------------------------------------------------------------------------
AUGMENTATION_RATIOS = [0.10, 0.25, 0.50]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
SEED = 42

LABEL_NAMES = {"negative": 0, "neutral": 1, "positive": 2}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GENERATED_DIR = DATA_DIR / "generated"

# ---------------------------------------------------------------------------
# Dataset (local file)
# Format per line: sentence@label  (label = positive | neutral | negative)
# file at: data/raw/Sentences_75Agree.txt
# ---------------------------------------------------------------------------
DATASET_FILE = RAW_DIR / "Sentences_75Agree.txt"

# ---------------------------------------------------------------------------
# Ollama generation settings
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
MAX_RETRIES = 3
TEMPERATURE = 0.8
MAX_TOKENS = 120
