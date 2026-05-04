# core/watermark.py

import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import binary_dilation
from sklearn.linear_model import OrthogonalMatchingPursuit
from utils.chaos import SecureChaos
import config

class CompressiveWatermark:
    def __init__(self, password=config.SECRET_PASSWORD):
        self.bs = config.BLOCK_SIZE
        self.crypto = SecureChaos(password)

  

    def _allocate(self, img, total_blocks):
        # Calculates block variances and allocates CS budget
        h, w = img.shape
        variances = [np.var(img[i:i+self.bs, j:j+self.bs]) 
                     for i in range(0, h, self.bs) for j in range(0, w, self.bs)]
        
        allocations = np.zeros(total_blocks, dtype=int)
        sorted_vars = np.sort(variances)
        thresh_smooth = sorted_vars[int(total_blocks * 0.40)]
        thresh_texture = sorted_vars[int(total_blocks * 0.80)]

        for i in range(total_blocks):
            if variances[i] <= thresh_smooth: allocations[i] = 8
            elif variances[i] > thresh_texture: allocations[i] = 32
            else: allocations[i] = 16

        factor = (total_blocks * config.TARGET_AVG_MEASUREMENTS) / np.sum(allocations)
        return np.round(allocations * factor).astype(int)
    


    def _calc_parity(self, array):
        parity = np.zeros_like(array, dtype=np.uint8)
        # Use top 7 bits (bits 1-7) to allow for 50+ dB PSNR
        for i in range(1, 8):
            parity ^= ((array >> i) & 1)
        return parity

    def embed(self, img):
        h, w = img.shape
        total_blocks = (h // self.bs) * (w // self.bs)
        allocations = self._allocate(img, total_blocks)
        
        cs_payloads = []
        block_idx = 0
        
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block = img[i:i+self.bs, j:j+self.bs]
                dct_vec = dctn(block, norm='ortho').flatten()
                
                dc = int(np.clip(np.round(dct_vec[0] / 8.0), 0, 255))
                ac = dct_vec[1:]
                
                num_meas = allocations[block_idx]
                Phi = self.crypto.generate_sensing_matrix(num_meas, block_idx)
                y_ac = np.dot(Phi, ac) if Phi is not None else np.array([])
                
                cs_payloads.append({'dc': dc, 'ac': y_ac})
                block_idx += 1

        P = self.crypto.generate_permutation(total_blocks)
        distributed_payloads = [cs_payloads[P[i]] for i in range(total_blocks)]

        # ── QUALITY & SECURITY FIX ──
        # 1. Use 0xFE to only modify the 1st LSB (Achieves 51+ dB PSNR)
        top7 = img & 0xFE
        
        # 2. XOR with Spatial Mask to defeat Copy-Move
        chaotic_mask = self.crypto.generate_binary_mask((h, w))
        auth_bit = self._calc_parity(top7) ^ chaotic_mask
        
        watermarked = top7 | auth_bit
        
        return watermarked, distributed_payloads, P, allocations

    def recover(self, tampered_img, payloads, P, allocations):
        h, w = tampered_img.shape
        recovered_img = tampered_img.copy().astype(np.float32)
        P_inv = np.argsort(P)

        # ── 1. SPATIALLY AWARE CRYPTOGRAPHIC TAMPER DETECTION ──
        top7 = tampered_img & 0xFE
        chaotic_mask = self.crypto.generate_binary_mask((h, w))
        
        # XORing with the chaotic mask catches displaced Copy-Move patches!
        expected_auth = self._calc_parity(top7) ^ chaotic_mask
        actual_auth = tampered_img & 1
        
        # Raw map (Has a 50% false-negative rate on tampered pixels)
        raw_tamper_map = (expected_auth != actual_auth).astype(np.uint8)
        
        # ── 2. IEEE FIX: MORPHOLOGICAL MASK SOLIDIFICATION ──
        # Close bridges the 50% gaps for solid attacks (Crop/Copy-Move)
        kernel_close = np.ones((5, 5), np.uint8)
        solid_tamper_map = cv2.morphologyEx(raw_tamper_map, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate slightly to ensure we swallow the S&P pixels that accidentally passed parity
        kernel_sp = np.ones((3, 3), np.uint8)
        dilated_map = cv2.dilate(solid_tamper_map, kernel_sp, iterations=1)
        
        pixel_tamper_map = dilated_map.astype(bool)

        # ── 3. OMP RECOVERY ──
        block_idx = 0
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block_tamper_mask = pixel_tamper_map[i:i+self.bs, j:j+self.bs]

                if np.any(block_tamper_mask):
                    idx = P_inv[block_idx]
                    payload = payloads[idx]
                    num_meas = allocations[block_idx]

                    dc_rec = payload['dc'] * 8.0
                    ac_rec = np.zeros(63)
                    if num_meas > 0 and len(payload['ac']) > 0:
                        Phi = self.crypto.generate_sensing_matrix(num_meas, block_idx)
                        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(1, num_meas // config.SPARSITY_RATIO))
                        try:
                            omp.fit(Phi, payload['ac'])
                            ac_rec = omp.coef_
                        except: pass

                    dct_vec = np.zeros(64)
                    dct_vec[0] = dc_rec
                    dct_vec[1:] = ac_rec
                    pixel_block = np.clip(idctn(dct_vec.reshape(self.bs, self.bs), norm='ortho'), 0, 255)

                    # ── FIX: Per-pixel blending ──
                    pixel_mask = block_tamper_mask.astype(np.float32)   
                    original_block = recovered_img[i:i+self.bs, j:j+self.bs]
                    blended = (pixel_mask * pixel_block + (1.0 - pixel_mask) * original_block)
                    recovered_img[i:i+self.bs, j:j+self.bs] = blended
                block_idx += 1

        recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)

        # ── FIX: Gaussian feather on boundary ring ──
        ring_kernel = np.ones((9, 9), dtype=bool)
        dilated_boundary = binary_dilation(pixel_tamper_map, structure=ring_kernel)
        boundary_ring = dilated_boundary & ~pixel_tamper_map         

        if boundary_ring.any():
            blurred = cv2.GaussianBlur(recovered_img, (7, 7), 2)
            alpha = 0.5
            recovered_img[boundary_ring] = (
                alpha * blurred[boundary_ring].astype(np.float32) + 
                (1.0 - alpha) * recovered_img[boundary_ring].astype(np.float32)
            ).astype(np.uint8)

        return recovered_img, pixel_tamper_map

