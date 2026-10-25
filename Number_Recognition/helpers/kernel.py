import numpy as np

def gaussian_kernel(A, B, h=1.0):
    A_sq = np.sum(A ** 2, axis=1)
    B_sq = np.sum(B ** 2, axis=1)
    dist_sq = A_sq[:, None] + B_sq[None, :] - 2 * (A @ B.T)
    dist_sq = np.maximum(dist_sq, 0)
    return np.exp(-dist_sq / (2 * h ** 2))