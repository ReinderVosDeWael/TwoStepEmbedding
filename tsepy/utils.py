"""Utilities for the TwoStepEmbedding class"""
import warnings
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from brainspace.gradient.kernels import compute_affinity


def nystrom(out_sample, reference_sample, reference_evec, kernel="cosine"):
    """Adds out of sample datapoints to a manifold using Nystrom extension.

    Parameters
    ----------
    out_sample : array-like
        Feature-by-sample-by-subject out-of-sample data.
    reference_sample : array-like
        Feature-by-sample-by-subject reference data.
    reference_evec : array-like
        Feature-by-component(-by-subject) reference eigenvectors. If no subject
        dimension is provided, then all subjects are projected onto the same
        manifold.
    kernel : str
        Kernel used for the affinity computation. For a list of valid kernels see
        BrainSpace's compute_affinity, defaults to "cosine".

    References
    ----------
    Gao, S., Mishne, G., & Scheinost, D. (2020). Non-linear manifold learning in
    fMRI uncovers a low-dimensional space of brain dynamics. bioRxiv.
    """

    out_sample = np.atleast_3d(out_sample)
    reference_sample = np.atleast_3d(reference_sample)
    reference_evec = np.atleast_3d(reference_evec)

    out_sample_manifold = np.zeros(
        (out_sample.shape[0], reference_evec.shape[1], out_sample.shape[2])
    )
    for i in range(out_sample.shape[0]):
        for j in range(out_sample.shape[2]):
            out_sample_vec = np.atleast_2d(out_sample[i, :, j])
            merged_data = np.concatenate(
                (out_sample_vec, reference_sample[:, :, j]), axis=0
            )
            affinity = compute_affinity(
                merged_data, kernel=kernel, sparsity=0, non_negative=False
            )[0, 1:]
            affinity = np.expand_dims(affinity, axis=1) / np.sum(affinity)

            out_sample_manifold[i, :, j] = np.sum(
                reference_evec[:, :, j] * affinity, axis=0
            )
    return out_sample_manifold


def optimize_kmeans(M, klim, random_state=None):
    """Optimizes a k-means clustering using the Calinski-Harabasz criterion.

    Parameters
    ----------
    M : array-like
        Array (samples, features) to be clustered.
    klim : array-like
        2-element array containing the minimm and maximum k.
    random_state : int, RandomState instance, None
        Random state initialization for k-means, by default None.

    Returns
    -------
    labels : numpy.array
        Vector of label assignments.

    References
    ----------
    Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
    Communications in Statistics-theory and Methods, 3(1), 1-27.
    """

    if klim[1] < klim[0]:
        ValueError("kmin must be smaller than kmax.")

    ks = range(klim[0], klim[1] + 1)
    score = -np.inf

    for i in range(len(ks)):
        kmeans_model = KMeans(n_clusters=ks[i], random_state=random_state)
        kmeans_model = kmeans_model.fit(M)
        new_labels = kmeans_model.labels_
        new_score = calinski_harabasz_score(M, new_labels)
        if new_score > score:
            score = new_score
            labels = new_labels
            k_opt = ks[i]

    return labels, k_opt, score


def brainsync(timeseries, reference=None):
    """Synchronizes timeseries based on SVD rotation.

    Parameters
    ----------
    timeseries : array-like
        Time-by-region-by-scan array of timeseries.
    reference : array-like, optional
        Time-by-region matrix to use as reference. If no reference and more than
        two scans are provided then the scan most similar to all others is used
        as reference. If no reference and only two scans are provided, then the
        first scan is used as reference. Defaults to None.

    Returns
    -------
    Y : numpy.array
        Time-by-region-by-scan array of syncronized timeseries.
    R : numpy.array
        Time-by-time-by-scan array of rotation matrices.

    References
    ----------
    Joshi, A. A., Chong, M., Li, J., Choi, S., & Leahy, R. M. (2018). Are you
    thinking what I'm thinking? Synchronization of resting fMRI time-series
    across subjects. NeuroImage, 172, 740-752.
    """

    n_time, n_roi, n_scans = timeseries.shape
    if n_roi > n_time:
        warnings.warn("There should be more regions (columns) than timepoints (rows).")

    if reference is None:
        if n_scans == 1:
            ValueError(
                "You must either provide a reference or multiple sets of timeseries."
            )
        elif n_scans == 2:
            idx = 0
        else:
            idx = find_central_scan(timeseries)
        reference = timeseries[:, :, idx]

    # Initialize arrays.
    rotated_ts = np.zeros((n_time, n_roi, n_scans))
    rotations = np.zeros((n_roi, n_roi, n_scans))

    # Rotate the data.
    for i in range(n_scans):
        ts_tmp, rotations[:, :, i] = rotate_data(timeseries[:, :, i].T, reference.T)
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
    Output data is normalized to zero mean and unit variance.
    """

    # Ascertain unit variance and zero mean.
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    reference = (reference - np.mean(reference, axis=0)) / np.std(reference, axis=0)

    U, _, V = np.linalg.svd(reference @ data.T)
    rotation_matrix = U @ V
    rotated_data = rotation_matrix @ data
    return rotated_data, rotation_matrix


def find_central_scan(timeseries):
    """Finds the scan most similar to all other scans based on minimum Euclidean
    distance between datapoints.

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
