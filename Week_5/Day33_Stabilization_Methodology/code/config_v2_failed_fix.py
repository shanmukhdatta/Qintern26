"""
config.py
Central configuration for the QGAN augmentation pipeline on CIC-MalMem-2022.
Edit values here rather than inside the other modules.
"""

import os

# ---------------- Paths ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(PROJECT_DIR)
DATA_PATH = os.path.join(WORK_DIR, "data", "malmem_original_reconstructed.csv")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs_v2")
LOG_DIR = os.path.join(WORK_DIR, "logs_v2")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

AUGMENTED_CSV_PATH = os.path.join(OUTPUT_DIR, "malmem_qgan_augmented.csv")
FIDELITY_TABLE_PATH = os.path.join(OUTPUT_DIR, "fidelity_metrics.csv")
DOWNSTREAM_TABLE_PATH = os.path.join(OUTPUT_DIR, "downstream_classifier_results.csv")

# ---------------- Columns ----------------
TYPE_LABEL_COL = "MalwareType"   # derived 4-class label: Benign/Ransomware/Spyware/Trojan
LABEL_COL = TYPE_LABEL_COL        # the label the QGAN pipeline augments/evaluates against
FAMILY_LABEL_COL = "Category"     # original fine-grained family name, kept for reference
BINARY_LABEL_COL = "Class"        # Benign / Malware
DROP_COLS = ["Filename"]           # non-feature columns to drop outright

# ---------------- Feature selection / dimensionality ----------------
N_SELECT_FEATURES = 12   # top-k features kept by SelectKBest before PCA
N_QUBITS = 6              # PCA target dimension == number of qubits
RANDOM_STATE = 42

# ---------------- Which classes get QGAN-augmented ----------------
# Benign is already the majority class and is left untouched.
EXCLUDE_FROM_AUGMENTATION = ["Benign"]

# Target sample count per augmented (malware family) class.
# None => match the largest existing malware family class count.
TARGET_SAMPLES_PER_CLASS = None

# ---------------- Quantum generator ----------------
N_VARIATIONAL_LAYERS = 6   # bumped from 4 -- untested fix for capacity mismatch
                            # (LR decay + 0.20 dropout were already active and insufficient)
LATENT_DIM = 6            # dimension of latent noise vector z
GEN_HIDDEN_DIM = 16       # classical front-end hidden width

# ---------------- Discriminator ----------------
DISC_HIDDEN_1 = 16
DISC_HIDDEN_2 = 8
LEAKY_RELU_ALPHA = 0.2
DISC_DROPOUT_P = 0.30   # raised further from 0.20 -- 0.20 was already insufficient

# ---------------- Training ----------------
EPOCHS = 100          # bumped from 30 -> 100 (README §7 suggested 60-100)
CHECKPOINT_EVERY = 5  # epochs between checkpoint saves (crash-safe resume)
BATCH_SIZE = 128
LR_GEN = 7e-3
LR_DISC = 2e-3
LR_DECAY = True           # NEW: cosine decay both LRs over training to fight the
LR_MIN_FRAC = 0.15        # growing late-training oscillation seen in the stability diagnosis
GRAD_CLIP_NORM = 1.0

# Stabilization terms (Q-SYNTH recipe)
INSTANCE_NOISE_BASE = 0.05
LABEL_SMOOTHING_GAMMA = 0.9
FEATURE_MATCHING_WEIGHT = 0.10
MOMENT_MATCH_ALPHA = 0.05   # mean-matching weight
MOMENT_MATCH_BETA = 0.03    # std-matching weight

EVAL_EVERY_N_EPOCHS = 10

# ---------------- Evaluation ----------------
DOWNSTREAM_TEST_SIZE = 0.25
