# run_experiments.py

import os
import glob
import cv2
import numpy as np
import pandas as pd
import warnings
from core.watermark import CompressiveWatermark
from core.attacks import Attacks
from utils.metrics import calc_psnr, calc_ssim, calc_nc, calc_tamper_metrics
from utils.pdf_report import build_visual_pdf
import config

warnings.filterwarnings("ignore")

def main():
    image_paths = glob.glob(os.path.join(config.DATASET_DIR, "*.*"))
    if not image_paths:
        print(f"No images found in {config.DATASET_DIR}. Please add images.")
        return

    system = CompressiveWatermark()
    results = []

    attacks = {
        "Crop 33%": lambda img: Attacks.crop(img, ratio=0.33),
        "Copy-Move": Attacks.copy_move,
        "Salt_and_Pepper_1%": lambda img: Attacks.salt_and_pepper(img, prob=0.01),
        "JPEG_80": lambda img: Attacks.jpeg_compression(img, quality=80)
    }

    for img_path in image_paths:
        # File parsing cleanup
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing Image: {base_name}")
        
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None: continue
        original = cv2.resize(original, (256, 256))

        # TRUE SELF-RECOVERY: Payload is embedded entirely inside the image
        watermarked = system.embed(original)
        psnr_w = calc_psnr(original, watermarked)
        ssim_w = calc_ssim(original, watermarked)

        for attack_name, attack_fn in attacks.items():
            tampered, gt_mask = attack_fn(watermarked)
            recovered, pred_mask = system.recover(tampered)

            psnr_r = calc_psnr(original, recovered)
            ssim_r = calc_ssim(original, recovered)
            nc_r = calc_nc(original, recovered)
            
            # EVALUATION FIX: JPEG is a global attack, Pixel-level TPR/FPR is undefined
            if "JPEG" in attack_name:
                tpr, fpr, f1 = np.nan, np.nan, np.nan
            else:
                tpr, fpr, f1 = calc_tamper_metrics(gt_mask, pred_mask)

            safe_attack_name = attack_name.replace(" ", "_").replace("%", "")
            prefix = os.path.join(config.REPORT_IMG_DIR, f"{base_name}_{safe_attack_name}")
            
            paths = {
                'orig': f"{prefix}_1_orig.png",
                'watermarked': f"{prefix}_2_wm.png",
                'tampered': f"{prefix}_3_attack.png",
                'true_map': f"{prefix}_4_truemap.png",
                'recovered': f"{prefix}_5_rec.png"
            }
            
            cv2.imwrite(paths['orig'], original)
            cv2.imwrite(paths['watermarked'], watermarked)
            cv2.imwrite(paths['tampered'], tampered)
            cv2.imwrite(paths['true_map'], (gt_mask * 255).astype(np.uint8)) 
            cv2.imwrite(paths['recovered'], recovered)

            results.append({
                "Image": base_name,
                "Attack": attack_name,
                "W-PSNR": round(psnr_w, 2),
                "W-SSIM": round(ssim_w, 4),
                "R-PSNR": round(psnr_r, 2),
                "R-SSIM": round(ssim_r, 4),
                "R-NC": round(nc_r, 4),
                "TPR": round(tpr, 4) if not np.isnan(tpr) else "N/A",
                "FPR": round(fpr, 4) if not np.isnan(fpr) else "N/A",
                "F1-Score": round(f1, 4) if not np.isnan(f1) else "N/A",
                "paths": paths
            })

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'paths'} for r in results])
    df.to_csv(config.CSV_REPORT, index=False)
    
    print("Generating Visual PDF Report...")
    build_visual_pdf(results, config.PDF_REPORT_PATH)
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE. AVERAGE RESULTS ACROSS ALL IMAGES:")
    print("="*60)
    
    # Clean and explicit Pandas aggregation calculation
    metric_cols = ["W-PSNR", "W-SSIM", "R-PSNR", "R-SSIM", "R-NC", "TPR", "FPR", "F1-Score"]
    df_summary = df.copy()
    
    # Replace the strings and explicitly cast ONLY the numerical columns to float
    df_summary[metric_cols] = df_summary[metric_cols].replace("N/A", np.nan).astype(float)
    
    # Group by attack and calculate mean
    summary = df_summary.groupby("Attack")[metric_cols].mean().round(4)
    print(summary)

if __name__ == "__main__":
    main()