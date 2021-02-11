import numpy as np
import sklearn as skl
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity
from .utils import brainsync
from sklearn.base import BaseEstimator


class TwoStepEmbedding(BaseEstimator):
    """Two Step Embedding"""

    def __init__(self, kernel="cosine", approach="dm", rng=0, d1=10, d2=10, klim=None):
        """Constructor for two step embedding.

        Parameters
        ----------
        kernel : str, optional
            Kernel for gradient computation, by default "cosine".
        approach : str, optional
            Dimensionality reduction for gradient computation, by default "dm".
        rng : int, optional
            Initialization of random state, by default 0.
        d1 : int, optional
            Number of omponents to compute in the first embedding step, by
            default 10.
        d2 : int, optional
            Number of components to copmute in the second embedding step, by
            default 10.
        """

        self.kernel = kernel
        self.approach = approach
        self.rng = rng
        self.d1 = d1
        self.d2 = d2
        self.klim = None

    def fit(self, timeseries, out_sample=None):
        """Fits the TwoStepEmbedding model

        Parameters
        ----------
        timeseries : array-like
            Region-by-time-by-subject timeseries array.
        out_sample : array-like, optional.
            Region-by-time-by-subject timeseries array of out-of-sample
            resting-state data, by default None.
        """

        # Run the embedding.
        self.psi_2, self.psi_1 = self.embedding(timeseries)

        # Insert out-of-sample data
        if out_sample is not None:
            if out_sample.shape[2] != 1:
                out_sample = brainsync(out_sample)
            psi_1_hat = self.nystrom(out_sample, timeseries, self.psi_1)
            psi_2_hat = self.nystrom(psi_1_hat, self.psi_1, self.psi_2)
            self.psi_2 = np.concatenate((self.psi_2, psi_2_hat), axis=0)

        # Run k-means clustering.
        if self.klim is not None:
            self.labels = self.optimize_kmeans(self)

    def embedding(self, timeseries, sparsity=0):
        """Performs a dual embedding

        Parameters
        ----------
        timeseries : array-like
            Region-by-time-by-subject timeseries array.

        Returns
        -------
        numpy.array
            Eigenvectors of the second embedding.
        numpy.array
            Eigenvectors of the first embedding.
        """
        # Step 1: Run GradientMaps for each subject.
        t, _, s = timeseries.shape
        psi_1 = np.zeros((t, self.d1, s))
        for i in range(s):
            gm = GradientMaps(
                n_components=self.d1,
                random_state=self.rng,
                kernel=self.kernel,
                approach=self.approach,
            )
            gm.fit(timeseries[:, :, i], sparsity=sparsity)
            psi_1[:, :, i] = gm.gradients_

        # Step 2: Run GradientMaps for the group-level gradients.
        gm = GradientMaps(
            n_components=self.d2,
            random_state=self.rng,
            kernel=self.kernel,
            approach=self.approach,
        )
        gm.fit(np.reshape(psi_1, (t, -1), order="F"), sparsity=sparsity)
        psi_2 = gm.gradients_
        return psi_2, psi_1

    def nystrom(self, out_sample, reference_sample, reference_evec):
        """Adds out of sample datapoints to a manifold.

        Parameters
        ----------
        out_sample : array-like
            Out-of-sample data.
        reference_sample : array-like
            Reference data.
        reference_evec : array-like
            Reference eigenvectors.
        kernel : str
            Kernel used for affinity computation.

        References
        ----------
        Gao, S., Mishne, G., & Scheinost, D. (2020). Non-linear manifold
        learning in fMRI uncovers a low-dimensional space of brain dynamics.
        bioRxiv.
        """
        y_k = np.zeros((out_sample.shape[0], reference_evec.shape[1]))
        for i in range(out_sample.shape[0]):
            K = compute_affinity(
                np.vstack(out_sample[i, :], reference_sample),
                kernel=self.kernel,
                sparsity=0,
                non_negative=False,
            )[1, :]
            y_k[i, :] = np.sum(reference_evec[i, :] * K)
        return y_k

    def optimize_kmeans(self, M):
        """Optimizes a k-means clustering using the Calinski-Harabasz criterion.

        Parameters
        ----------
        M : array-like
            Array (samples, features) to be clustered.

        Returns
        -------
        labels : numpy.array
            Vector of label assignments.

        References
        ----------
        Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
        Communications in Statistics-theory and Methods, 3(1), 1-27.
        """

        if self.klim[1] < self.klim[0]:
            ValueError("kmin must be smaller than kmax.")

        ks = range(self.klim[0], self.klim[1] + 1)
        score = -np.inf

        for i in ks:
            kmeans_model = skl.cluster.KMeans(n_clusters=ks[i], random_state=self.rng)
            kmeans_model = kmeans_model.fit(M)
            new_labels = kmeans_model.labels_
            new_score = skl.metrics.calinski_harabasz_score(M, new_labels)
            if new_score > score:
                score = new_score
                labels = new_labels
                k_opt = ks[i]

        return labels, k_opt, score

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, x):
        valid_kernels = [
            "pearson",
            "spearman",
            "cosine",
            "normalized_angle",
            "gaussian",
        ]
        if x not in valid_kernels and x is not None:
            ValueError(
                "Kernel must be None or one of the following: " ", ".join(valid_kernels)
            )
        self._kernel = x

    @property
    def approach(self):
        return self._approach

    @approach.setter
    def approach(self, x):
        valid_approaches = ["pca", "le", "dm"]
        if x not in valid_approaches:
            ValueError(
                "Approach must be one of the following:  + " ", ".join(valid_approaches)
            )
        self._approach = x

    @property
    def d1(self):
        return self._d1

    @d1.setter
    def d1(self, x):
        if not np.isscalar(x):
            ValueError("d1 must be a scalar.")
        self._d1 = x

    @property
    def d2(self):
        return self._d1

    @d2.setter
    def d2(self, x):
        if not np.isscalar(x):
            ValueError("d1 must be a scalar.")
        self._d2 = x
