"""Unit tests for utils functions."""
import numpy as np
from tsepy.utils import brainsync, find_central_scan, rotate_data


def test_brainsync():
    """Tests whether the outputs of tsepy.utils.brainsync have te correct shape
    and whether the within-matrix correlations of the input and output matrix
    remain identical."""
    x1 = np.random.rand(10, 5, 3)
    y1, rotations = brainsync(x1)
    y2, _ = brainsync(x1[:, :, 1:], x1[:, :, 0])
    for i in range(x1.shape[2]):
        assert np.allclose(np.corrcoef(x1[:, :, i]), np.corrcoef(y1[:, :, i]))
        if i != 0:
            assert np.allclose(np.corrcoef(x1[:, :, i]), np.corrcoef(y2[:, :, i - 1]))
    assert rotations.shape == (x1.shape[1], x1.shape[1], x1.shape[2])


def test_find_central_scan():
    """Tests whether the output of tsepy.utils.find_central_scan is less than
    its length."""
    x = np.random.rand(10, 10, 3)
    idx = find_central_scan(x)
    assert idx < x.shape[2]


def test_rotate_data():
    """Tests whether the within-matrix correlation before and after rotation
    with tsepy.utils.rotate_data remain identical"""
    data = np.random.rand(10, 5)
    reference = np.random.rand(10, 5)

    rotated_data, rotation_matrix = rotate_data(data, reference)

    r1 = np.corrcoef(data.T)
    r2 = np.corrcoef(rotated_data.T)
    assert np.allclose(r1, r2)
