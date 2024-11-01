import numpy as np
from config import get_similarity_measure

def compute_similarity(x: np.ndarray, y: np.ndarray) -> float:
    similarity_measure = get_similarity_measure()

    if similarity_measure == "cosine":
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    elif similarity_measure == "dot":
        return np.dot(x, y)
    else:
        raise ValueError(f"Invalid similarity measure: {similarity_measure}")
