# utils/chaos.py

import hashlib
import numpy as np

class SecureChaos:
    def __init__(self, password: str):
        self.password = password
        # Generate a deterministic 256-bit hash from the password
        hash_hex = hashlib.sha256(password.encode()).hexdigest()
        
        # Use parts of the hash to set the Logistic Map initial state (x0) and parameter (r)
        # x0 must be in (0, 1) and r in (3.9, 4.0) for deep chaos
        self.x0 = (int(hash_hex[:16], 16) / (16**16)) * 0.9 + 0.05
        self.r = 3.9 + (int(hash_hex[16:32], 16) / (16**16)) * 0.1
        self.master_seed = int(hash_hex[32:48], 16)

    def logistic_map_sequence(self, length: int) -> np.ndarray:
        """Generates a highly sensitive chaotic sequence."""
        seq = np.zeros(length)
        x = self.x0
        for i in range(length):
            x = self.r * x * (1 - x)
            seq[i] = x
        return seq

    def generate_permutation(self, length: int) -> np.ndarray:
        """Creates a secure permutation matrix P using sorted chaotic sequences."""
        chaotic_seq = self.logistic_map_sequence(length)
        return np.argsort(chaotic_seq)

    def generate_sensing_matrix(self, num_measurements: int, block_idx: int) -> np.ndarray:
        """Generates the Phi matrix using a secure PRNG seeded by the master hash + block index."""
        if num_measurements == 0:
            return None
        # Using NumPy's modern Generator with PCG64 for rigorous statistical randomness
        seed = (self.master_seed + block_idx) % (2**32)
        rng = np.random.default_rng(seed)
        
        Phi = rng.standard_normal((num_measurements, 63))
        Q, _ = np.linalg.qr(Phi.T)
        return Q.T
    
    def generate_binary_mask(self, shape) -> np.ndarray:
        """Generates a spatial chaotic mask to defeat Copy-Move attacks."""
        length = shape[0] * shape[1]
        seq = self.logistic_map_sequence(length)
        # Convert chaotic floats into binary 0 or 1
        return (seq > 0.5).astype(np.uint8).reshape(shape)