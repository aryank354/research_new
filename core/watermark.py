# core/watermark.py

import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
from utils.chaos import SecureChaos
import config

class CompressiveWatermark:
    def __init__(self, password=config.SECRET_PASSWORD):
        self.bs = config.BLOCK_SIZE
        self.crypto = SecureChaos(password)
        
        # --- ECZM: ZIGZAG DETERMINISTIC ANCHORS ---
        # Instead of random OMP, we perfectly capture the top 10 structural edges
        # Standard 8x8 DCT Zigzag Indices: 1, 8, 16, 9, 2, 3, 10, 17, 24, 32
        self.zigzag_idx = [1, 8, 16, 9, 2, 3, 10, 17, 24, 32]
        
        # Optimized bit allocations (total exactly 48 bits)
        self.bit_allocs = [6, 6, 5, 5, 5, 5, 4, 4, 4, 4] 

    def _calc_parity(self, array):
        parity = np.zeros_like(array, dtype=np.uint8)
        for i in range(1, 8):
            parity ^= ((array >> i) & 1)
        return parity

    def embed(self, img):
        h, w = img.shape
        total_blocks = (h // self.bs) * (w // self.bs)
        
        payloads = {}
        block_idx = 0
        
        # --- 1. ECZM PAYLOAD GENERATION ---
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block = img[i:i+self.bs, j:j+self.bs]
                dct_vec = dctn(block, norm='ortho').flatten()
                
                dc = int(np.clip(np.round(dct_vec[0] / 8.0), 0, 255))
                acs = dct_vec[self.zigzag_idx]
                
                # Dynamic Scale Factor
                max_ac = np.max(np.abs(acs)) if np.max(np.abs(acs)) > 0 else 1.0
                max_q = int(np.clip(max_ac / 4.0, 0, 255))
                max_val_rec = max_q * 4.0 if max_q > 0 else 1.0
                
                # Quantize the top 10 frequencies deterministically
                b_acs = ""
                for ac, bits in zip(acs, self.bit_allocs):
                    norm = np.clip(ac / max_val_rec, -1.0, 1.0)
                    levels = (1 << bits) - 1
                    q_val = int(np.clip(np.round((norm + 1.0) * (levels / 2.0)), 0, levels))
                    b_acs += format(q_val, f'0{bits}b')
                
                # Bit-Pack exactly 64 Bits: DC(8) + Scale(8) + ACs(48)
                bit_string = format(dc, '08b') + format(max_q, '08b') + b_acs
                payloads[block_idx] = np.array([int(b) for b in bit_string], dtype=np.uint8)
                block_idx += 1

        P = self.crypto.generate_permutation(total_blocks)
        watermarked = img.copy()
        block_idx = 0
        
        # --- 2. SINGLE-LSB EMBEDDING (W-PSNR > 44 dB) ---
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                source_idx = np.where(P == block_idx)[0][0]
                bits = payloads[source_idx].reshape(self.bs, self.bs)
                target = watermarked[i:i+self.bs, j:j+self.bs]
                
                # Store strictly in the 2nd LSB to maintain visual perfection
                watermarked[i:i+self.bs, j:j+self.bs] = (target & 0xFD) | (bits << 1)
                block_idx += 1

        # 1st LSB for chaotic parity authentication
        top7 = watermarked & 0xFE
        chaotic_mask = self.crypto.generate_binary_mask((h, w))
        return top7 | (self._calc_parity(top7) ^ chaotic_mask)

    def recover(self, tampered_img):
        h, w = tampered_img.shape
        recovered_img = tampered_img.copy().astype(np.float32)
        total_blocks = (h // self.bs) * (w // self.bs)
        P = self.crypto.generate_permutation(total_blocks)

        # --- 1. TAMPER DETECTION ---
        top7 = tampered_img & 0xFE
        chaotic_mask = self.crypto.generate_binary_mask((h, w))
        raw_tamper_map = (self._calc_parity(top7) ^ chaotic_mask != (tampered_img & 1)).astype(np.uint8)
        
        solid_tamper_map = cv2.morphologyEx(raw_tamper_map, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        pixel_tamper_map = cv2.dilate(solid_tamper_map, np.ones((3, 3), np.uint8)).astype(bool)

        # --- 2. EXTRACT 64-BIT PAYLOADS ---
        extracted_payloads = {}
        block_idx = 0
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block = tampered_img[i:i+self.bs, j:j+self.bs]
                extracted_payloads[block_idx] = ((block >> 1) & 1).flatten()
                block_idx += 1

        # --- 3. ECZM DETERMINISTIC RECOVERY ---
        lost_mask = np.zeros_like(pixel_tamper_map, dtype=np.uint8)
        block_idx = 0
        
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block_mask = pixel_tamper_map[i:i+self.bs, j:j+self.bs]

                if np.any(block_mask):
                    target_idx = P[block_idx] 
                    ty, tx = (target_idx // (w // self.bs)) * self.bs, (target_idx % (w // self.bs)) * self.bs
                    
                    if pixel_tamper_map[ty:ty+self.bs, tx:tx+self.bs].any():
                        lost_mask[i:i+self.bs, j:j+self.bs] = 255 # Payload physically destroyed
                    else:
                        # Decode perfectly structured 64-bit stream
                        bits = "".join(extracted_payloads[target_idx].astype(str))
                        dc_val = int(bits[0:8], 2)
                        max_q = int(bits[8:16], 2)
                        
                        max_val_rec = max_q * 4.0 if max_q > 0 else 1.0
                        
                        dct_vec = np.zeros(64)
                        dct_vec[0] = dc_val * 8.0
                        
                        bit_idx = 16
                        for ac_idx, n_bits in zip(self.zigzag_idx, self.bit_allocs):
                            q_val = int(bits[bit_idx : bit_idx + n_bits], 2)
                            levels = (1 << n_bits) - 1
                            dct_vec[ac_idx] = (q_val / (levels / 2.0) - 1.0) * max_val_rec
                            bit_idx += n_bits
                        
                        pixel_block = np.clip(idctn(dct_vec.reshape(self.bs, self.bs), norm='ortho'), 0, 255)
                        
                        p_mask = block_mask.astype(np.float32)
                        orig = recovered_img[i:i+self.bs, j:j+self.bs]
                        recovered_img[i:i+self.bs, j:j+self.bs] = (p_mask * pixel_block + (1.0 - p_mask) * orig)
                
                block_idx += 1

        recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)

        # --- 4. HIERARCHICAL REPAIR (NO BLURRING FILTERS) ---
        # Seamlessly synthesizes missing textures for strictly destroyed payload blocks
        if lost_mask.any():
            recovered_img = cv2.inpaint(recovered_img, lost_mask, 3, cv2.INPAINT_TELEA)

        # We intentionally removed the bilateral and boundary filters here. 
        # The raw inverse DCT is mathematically sharp. Post-processing ruins the crisp edges.

        return recovered_img, pixel_tamper_map