# utils/chaos.py

import hashlib
import numpy as np

class SecureChaos:
    def __init__(self, password: str):
        self.password = password
        hash_hex = hashlib.sha256(password.encode()).hexdigest()
        
        # Logistic map parameters generated from cryptographic hash
        self.x0 = (int(hash_hex[:16], 16) / (16**16)) * 0.9 + 0.05
        self.r = 3.9 + (int(hash_hex[16:32], 16) / (16**16)) * 0.1

    def logistic_map_sequence(self, length: int) -> np.ndarray:
        """Generates a highly sensitive chaotic sequence."""
        seq = np.zeros(length)
        x = self.x0
        # Cryptographic warm-up to bypass transient phase
        for _ in range(200):  
            x = self.r * x * (1 - x)
            
        for i in range(length):
            x = self.r * x * (1 - x)
            seq[i] = x
        return seq

    def generate_permutation(self, length: int) -> np.ndarray:
        """Creates a secure permutation matrix P."""
        chaotic_seq = self.logistic_map_sequence(length)
        return np.argsort(chaotic_seq)
    
    def generate_binary_mask(self, shape) -> np.ndarray:
        """Generates a spatial chaotic mask for authentication XORing."""
        length = shape[0] * shape[1]
        seq = self.logistic_map_sequence(length)
        return (seq > 0.5).astype(np.uint8).reshape(shape)