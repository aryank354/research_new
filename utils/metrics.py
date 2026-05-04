# utils/metrics.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, confusion_matrix

def calc_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)

def calc_ssim(img1, img2):
    return ssim(img1, img2, data_range=255)

def calc_nc(img1, img2):
    """
    Zero-Mean Normalized Cross-Correlation (ZNCC).
    Subtracting the mean aligns with standard literature definitions.
    """
    img1 = img1.astype(float) - np.mean(img1)
    img2 = img2.astype(float) - np.mean(img2)
    
    numerator = np.sum(img1 * img2)
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    return numerator / (denominator + 1e-10)

def calc_tamper_metrics(true_mask, pred_mask):
    """Returns True Positive Rate, False Positive Rate, and F1-Score."""
    y_true = true_mask.flatten().astype(int)
    y_pred = pred_mask.flatten().astype(int)
    
    f1 = f1_score(y_true, y_pred, zero_division=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    tpr = tp / (tp + fn + 1e-10)  # Sensitivity
    fpr = fp / (fp + tn + 1e-10)  # False alarms
    return tpr, fpr, f1