# config.py

import os

# System Parameters
BLOCK_SIZE = 8
TARGET_AVG_MEASUREMENTS = 24
SPARSITY_RATIO = 2  # num_meas // 2
SOLID_ATTACK_THRESHOLD = 12

# File Paths
DATASET_DIR = "RawImages"
OUTPUT_DIR = "results"
CSV_REPORT = os.path.join(OUTPUT_DIR, "ablation_results.csv")

# Security
SECRET_PASSWORD = "IEEE_Journal_Strong_Key_2026!"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# config.py (add these lines at the bottom)

# PDF Reporting Paths
REPORT_IMG_DIR = os.path.join(OUTPUT_DIR, "report_images")
PDF_REPORT_PATH = os.path.join(OUTPUT_DIR, "Visual_Ablation_Report.pdf")

os.makedirs(REPORT_IMG_DIR, exist_ok=True)