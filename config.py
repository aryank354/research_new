# config.py

import os

# System Parameters
BLOCK_SIZE = 8
TARGET_AVG_MEASUREMENTS = 16
SPARSITY_RATIO = 2  # num_meas // 2
SOLID_ATTACK_THRESHOLD = 12

# File Paths
DATASET_DIR = "RawImages"
OUTPUT_DIR = "results"
CSV_REPORT = os.path.join(OUTPUT_DIR, "ablation_results.csv")

# Security
SECRET_PASSWORD = "IEEE_Journal_Strong_Key_2026!"

os.makedirs(OUTPUT_DIR, exist_ok=True)