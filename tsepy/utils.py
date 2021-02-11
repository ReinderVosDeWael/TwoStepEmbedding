"""Utilities for the TwoStepEmbedding class"""

import warnings
import numpy as np


def brainsync(timeseries, reference=None):
    """Syncronizes timeseries based on SVD rotation.

    Parameters
    ----------
    timeseries : array-like
        Time-by-region-by-subject array of timeseries.
    reference : array-like, optional
        Time-by-region matrix to use as reference, defaults to None.

    Returns
    -------
    Y : numpy.array
        Time-by-region-by-subject array of syncronized timeseries.
    R : numpy.array
        Time-by-time-by-subject array of rotation matrices.

    Refererences
    ------------
    Joshi, A. A., Chong, M., Li, J., Choi, S., & Leahy, R. M. (2018). Are you
    thinking what I'm thinking? Synchronization of resting fMRI time-series
    across subjects. NeuroImage, 172, 740-752.
    """

    n_time, n_roi, n_subjects = timeseries.shape
    if n_roi > n_time:
        warnings.warn("There should be more regions (columns) than timepoints (rows).")

    if reference is None:
        if n_subjects == 1:
            ValueError(
                "You must either provide a reference or multiple sets of timeseries."
            )
        elif n_subjects == 2:
            idx = 0
        else:
            idx = find_central_scan(timeseries)
        reference = timeseries[:, :, idx]

    # Initialize arrays.
    rotated_ts = np.zeros((n_time, n_roi, n_subjects))
    rotations = np.zeros((n_roi, n_roi, n_subjects))

    # Rotate the data.
    for i in range(n_subjects):
        ts_tmp, rotations[:, :, i] = rotate_data(
            timeseries[:, :, i].T, reference.T
        )
        rotated_ts[:, :, i] = ts_tmp.T
    return rotated_ts, rotations


def rotate_data(data, reference):
    """Rotates data to match the reference using SVD.

    Parameters
    ----------
    data : array-like
        Data (sample-by-feature) to be rotated.
    reference : array-like
        Reference dataset.

    Returns
    -------
    numpy.array
        Rotated data.
    numpy.array
        Rotation matrix.

    Notes
    -----
    Output data is normalized to zero mean unit variance.
    """

    # Ascertain unit variance and zero mean.
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    reference = (reference - np.mean(reference, axis=0)) / np.std(reference, axis=0)

    U, _, V = np.linalg.svd(reference @ data.T)
    rotation_matrix = U @ V
    rotated_data = rotation_matrix @ data
    return rotated_data, rotation_matrix


def find_central_scan(timeseries):
    """Finds the scan most similar to all others

    Parameters
    ----------
    timeseries : numpy.array
        Time-by-region-by-subject matrix of timeseries data.

    Returns
    -------
    numpy.array
        The scan most similar to all others.


    """
    n_subjects = timeseries.shape[2]
    D = np.zeros((n_subjects, n_subjects))
    for i in range(n_subjects - 1):
        for j in range(i, n_subjects):
            D[i, j] = np.mean(
                np.linalg.norm(timeseries[:, :, i] - timeseries[:, :, j], axis=1)
            )
    D = D + D.T
    idx = np.argmin(np.sum(D, axis=1))
    return idx
