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

    def _calc_parity(self, array):
        parity = np.zeros_like(array, dtype=np.uint8)
        for i in range(3, 8):
            parity ^= ((array >> i) & 1)
        return parity

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

        # SECURE CHAOTIC PERMUTATION
        P = self.crypto.generate_permutation(total_blocks)
        distributed_payloads = [cs_payloads[P[i]] for i in range(total_blocks)]

        # AUTHENTICATION
        top5 = img & 0xF8
        watermarked = top5 | self._calc_parity(top5)
        
        return watermarked, distributed_payloads, P, allocations

    def recover(self, tampered_img, payloads, P, allocations):
        h, w = tampered_img.shape
        recovered_img = tampered_img.copy().astype(np.float32)
        P_inv = np.argsort(P)

        # 1. SMART TAMPER DETECTION
        top5 = tampered_img & 0xF8
        expected_auth = self._calc_parity(top5)
        raw_tamper_map = (expected_auth != (tampered_img & 1))
        
        pixel_tamper_map = np.zeros((h, w), dtype=bool)
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block_mask = raw_tamper_map[i:i+self.bs, j:j+self.bs]
                if np.sum(block_mask) > config.SOLID_ATTACK_THRESHOLD:
                    pixel_tamper_map[i:i+self.bs, j:j+self.bs] = True
                elif np.sum(block_mask) > 0:
                    pixel_tamper_map[i:i+self.bs, j:j+self.bs] = block_mask

        # 2. OMP RECOVERY
        block_idx = 0
        for i in range(0, h, self.bs):
            for j in range(0, w, self.bs):
                block_mask = pixel_tamper_map[i:i+self.bs, j:j+self.bs]
                if np.any(block_mask):
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

                    orig = recovered_img[i:i+self.bs, j:j+self.bs]
                    mask_float = block_mask.astype(np.float32)
                    recovered_img[i:i+self.bs, j:j+self.bs] = (mask_float * pixel_block + (1 - mask_float) * orig)
                block_idx += 1

        # 3. GAUSSIAN FEATHERING
        recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
        dilated = binary_dilation(pixel_tamper_map, structure=np.ones((9, 9), dtype=bool))
        ring = dilated & ~pixel_tamper_map
        if ring.any():
            blurred = cv2.GaussianBlur(recovered_img, (7, 7), 2)
            recovered_img[ring] = (0.5 * blurred[ring] + 0.5 * recovered_img[ring]).astype(np.uint8)

        return recovered_img, pixel_tamper_map