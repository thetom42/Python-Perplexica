import pytest
import numpy as np
from utils.compute_similarity import compute_similarity
from config import get_similarity_measure

@pytest.fixture
def mock_config(monkeypatch):
    def mock_get_similarity_measure():
        return "cosine"
    monkeypatch.setattr("backend.config.get_similarity_measure", mock_get_similarity_measure)

def test_compute_similarity_cosine(mock_config):
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])

    similarity = compute_similarity(x, y)
    expected_similarity = 1.0  # Cosine similarity of parallel vectors is 1

    assert np.isclose(similarity, expected_similarity)

def test_compute_similarity_dot_product(monkeypatch):
    def mock_get_similarity_measure():
        return "dot"
    monkeypatch.setattr("backend.config.get_similarity_measure", mock_get_similarity_measure)

    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])

    similarity = compute_similarity(x, y)
    expected_similarity = 28  # Dot product of [1,2,3] and [2,4,6]

    assert np.isclose(similarity, expected_similarity)

def test_compute_similarity_invalid_measure(monkeypatch):
    def mock_get_similarity_measure():
        return "invalid_measure"
    monkeypatch.setattr("backend.config.get_similarity_measure", mock_get_similarity_measure)

    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])

    with pytest.raises(ValueError):
        compute_similarity(x, y)

def test_compute_similarity_different_dimensions():
    x = np.array([1, 2, 3])
    y = np.array([2, 4])

    with pytest.raises(ValueError):
        compute_similarity(x, y)
