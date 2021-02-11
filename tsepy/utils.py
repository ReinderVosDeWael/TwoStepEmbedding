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

    n_roi, n_time, n_subjects = timeseries.shape()
    if n_roi > n_time:
        warnings.warn("There should be more regions (columns) than timepoints (rows).")

    if reference is None:
        if n_subjects == 1:
            ValueError(
                "You must either provide a reference or multiple sets of timeseries."
            )
        elif n_subjects == 2:
            reference = timeseries[:, :, 1]
        else:
            reference = find_central_scan(timeseries)

    # Initialize arrays.
    rotated_ts = np.zeros(n_roi, n_time, n_subjects)
    rotations = np.zeros(n_roi, n_roi, n_subjects)

    unitnorm_timeseries = timeseries / np.sqrt(np.sum(timeseries ** 2, axis=1))

    # Rotate the data.
    for i in range(n_subjects):
        rotated_ts[:, :, i], rotations[:, :, i] = rotate_data(
            unitnorm_timeseries[:, :, i], reference
        )
    return rotated_ts, rotations


def rotate_data(data, reference):
    """Rotates data to match the reference using SVD.

    Parameters
    ----------
    data : array-like
        Data to be rotated.
    reference : array-like
        Reference dataset.

    Returns
    -------
    numpy.array
        Rotated Data.
    numpy.array
        Rotations.
    """
    U, _, V = np.linalg.svd(reference * data.T)
    R = np.matmul(U, V.T)
    Ys = np.matmul(R, data)
    return Ys, R


def find_central_scan(timeseries):
    """Finds the scan most similar to all others

    Parameters
    ----------
    timeseries : numpy.array
        Time-by-region-by-subject matrix of timeseries data.

    Returns
    -------
    numpy.array
        The scan most similarto all others.


    """
    n_subjects = timeseries.shape[2]
    D = np.zeros((n_subjects, n_subjects))
    for i in range(n_subjects - 1):
        for j in range(i, n_subjects):
            D[i, j] = np.mean(
                vecnorm(timeseries[:, :, i] - timeseries[:, :, j], axis=1), axis=1
            )

    D = D + D.T
    idx = np.argmin(sum(D, axis=1))
    return timeseries[:, :, idx]


def vecnorm(x, axis=0):
    """Computes the vector norm.

    Parameters
    ----------
    x : array-like
        Matrix of which vector norms will be computed
    axis : int, optional
        Axis along which to compute vector norms, by default 0

    Returns
    -------
    numpy.array
        Vector norms.
    """
    return np.sqrt(np.sum(x ** 2, axis=axis))
