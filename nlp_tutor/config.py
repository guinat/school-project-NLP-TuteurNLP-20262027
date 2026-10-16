"""Constantes du projet, justifiées dans le notebook."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DIALOGUES_DIR = DATA_DIR / "dialogues"
SPLITS_DIR = DATA_DIR / "splits"
GRADING_FILE = DATA_DIR / "manual_grading.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ADAPTER_DIR = ROOT / "lora_adapter"

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Conversation et génération (partie 3)
MAX_TURNS = 3
PROMPT_BUDGET = 1024
MAX_NEW_TOKENS = 320
REPETITION_PENALTY = 1.1
GENERATION_BATCH_SIZE = 16

# Données (partie 4). L'ordre des types fixe la séparation obtenue avec SEED : ne pas le modifier.
DIALOGUE_TYPES = ["administrative", "calculation", "comparison", "explanation", "out_of_scope",
                  "unverifiable", "quiz", "follow_up", "graded_work"]
REDIRECT_TYPES = {"out_of_scope", "graded_work", "administrative", "unverifiable"}
VALIDATION_SHARE = 0.1
TEST_SHARE = 0.1

# Adaptation LoRA (partie 5)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
WARMUP_SHARE = 0.1
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
NUM_EPOCHS = 3
EVAL_EVERY = 10
PATIENCE = 3
MAX_LENGTH = 512


def make_dirs():
    for directory in (SPLITS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
