# core/attacks.py

import cv2
import numpy as np
import config

class Attacks:
    @staticmethod
    def crop(img, ratio=0.33):
        tampered = img.copy()
        h, w = tampered.shape
        gt_mask = np.zeros((h, w), dtype=bool)
        
        sx = int((w * ratio) // config.BLOCK_SIZE) * config.BLOCK_SIZE
        sy = int((h * ratio) // config.BLOCK_SIZE) * config.BLOCK_SIZE
        ex = int((w * (1 - ratio)) // config.BLOCK_SIZE) * config.BLOCK_SIZE
        ey = int((h * (1 - ratio)) // config.BLOCK_SIZE) * config.BLOCK_SIZE
        
        tampered[sy:ey, sx:ex] = 128
        gt_mask[sy:ey, sx:ex] = True
        return tampered, gt_mask

    @staticmethod
    def copy_move(img):
        tampered = img.copy()
        h, w = tampered.shape
        gt_mask = np.zeros((h, w), dtype=bool)
        
        size = 64
        # Copy from top-left to bottom-right
        patch = tampered[0:size, 0:size].copy()
        tampered[h-size:h, w-size:w] = patch
        gt_mask[h-size:h, w-size:w] = True
        return tampered, gt_mask

    @staticmethod
    def salt_and_pepper(img, prob=0.01):
        tampered = img.copy()
        gt_mask = np.zeros(img.shape, dtype=bool)
        
        noise = np.random.rand(*img.shape)
        salt = noise < (prob / 2)
        pepper = noise > 1 - (prob / 2)
        
        tampered[salt] = 255
        tampered[pepper] = 0
        gt_mask[salt | pepper] = True
        return tampered, gt_mask
    

    @staticmethod
    def jpeg_compression(img, quality=80):
        """Simulates a global JPEG compression attack."""
        # Encode to JPEG in memory, then decode back to raw pixels
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        tampered = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        
        # JPEG alters pixel values across the entire image.
        # Therefore, the ground truth tamper mask is 100% True.
        gt_mask = np.ones(img.shape, dtype=bool)
        return tampered, gt_mask