"""Methods for the TwoStepEmbedding class."""
import numpy as np
from brainspace.gradient import GradientMaps
from tsepy.utils import brainsync, nystrom, optimize_kmeans


class TwoStepEmbedding:
    """Two Step Embedding"""

    def __init__(
        self,
        kernel="cosine",
        approach="dm",
        random_state=0,
        sparsity=0,
        n_comp_1=10,
        n_comp_2=10,
        klim=None,
    ):
        """Constructor for two step embedding.

        Parameters
        ----------
        kernel : str, optional
            Kernel for gradient computation, by default "cosine".
        approach : str, optional
            Dimensionality reduction for gradient computation, by default "dm".
        random_state : int, optional
            Initialization of random state, by default 0.
        sparsity : int, optional
            Matrix sparsification for kernel computation. Value must be in the
            range [0,100), by default 0.
        n_comp_1 : int, optional
            Number of components to compute in the first embedding step, by
            default 10.
        n_comp_2 : int, optional
            Number of components to compute in the second embedding step, by
            default 10.
        klim : array-like, optional
            2-element vector containing the minimum and maximum k for the
            k-means optimization.

        References
        ----------
        Gao, S., Mishne, G., & Scheinost, D. (2020). Non-linear manifold
        learning in fMRI uncovers a low-dimensional space of brain dynamics.
        bioRxiv.
        """

        self.kernel = kernel
        self.approach = approach
        self.random_state = random_state
        self.sparsity = sparsity
        self.n_comp_1 = n_comp_1
        self.n_comp_2 = n_comp_2
        self.klim = klim

    def fit(self, timeseries, out_sample=None):
        """Fits the TwoStepEmbedding model

        Parameters
        ----------
        timeseries : array-like
            Time-by-region-by-subject timeseries array.
        out_sample : array-like, optional.
            Time-by-region-by-subject timeseries array of out-of-sample
            resting-state data, by default None.
        """

        # Run the embedding.
        self._embedding(timeseries)

        # Insert out-of-sample data
        if out_sample is not None:
            if out_sample.shape[2] != 1:
                out_sample, _ = brainsync(out_sample)
            psi_1_hat = nystrom(out_sample, timeseries, self.psi_1, self._kernel)

            psi_1_hat = np.reshape(psi_1_hat, (psi_1_hat.shape[0], -1), order="F")
            psi_1_rs = np.reshape(self.psi_1, (self.psi_1.shape[0], -1), order="F")

            psi_2_hat = nystrom(psi_1_hat, psi_1_rs, self.psi_2, self._kernel)
            self.psi_2 = np.concatenate((self.psi_2, np.squeeze(psi_2_hat)), axis=0)

        # Run k-means clustering.
        if self._klim is not None:
            self.labels = optimize_kmeans(self.psi_2, self._klim, self._random_state)

    def _embedding(self, timeseries):
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
        self.psi_1 = np.zeros((t, self._n_comp_1, s))
        for i in range(s):
            gm_1 = GradientMaps(
                n_components=self._n_comp_1,
                random_state=self._random_state,
                kernel=self._kernel,
                approach=self._approach,
            )
            gm_1.fit(timeseries[:, :, i], sparsity=self._sparsity)
            self.psi_1[:, :, i] = gm_1.gradients_

        # Step 2: Run GradientMaps for the subject-level gradients.
        gm_2 = GradientMaps(
            n_components=self._n_comp_2,
            random_state=self._random_state,
            kernel=self._kernel,
            approach=self._approach,
        )
        gm_2.fit(np.reshape(self.psi_1, (t, -1), order="F"), sparsity=self._sparsity)
        self.psi_2 = gm_2.gradients_

    @property
    def _kernel(self):
        """Kernel for gradient computation."""
        return self.kernel

    @_kernel.setter
    def _kernel(self, x):
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
        self.kernel = x

    @property
    def _approach(self):
        """Dimensionality reduction for gradient computation."""
        return self.approach

    @_approach.setter
    def _approach(self, x):
        valid_approaches = ["pca", "le", "dm"]
        if x not in valid_approaches:
            ValueError(
                "Approach must be one of the following:  + " ", ".join(valid_approaches)
            )
        self.approach = x

    @property
    def _random_state(self):
        """Initialization for the random state."""
        return self.random_state

    @_random_state.setter
    def _random_state(self, x):
        self.random_state = x

    @property
    def _sparsity(self):
        """Sparsity used in kernel computation."""
        return self.sparsity

    @_sparsity.setter
    def _sparsity(self, x):
        if x < 0 or x >= 100:
            ValueError("Sparsity must be in the range [0, 100).")
        self.sparsity = x

    @property
    def _n_comp_1(self):
        """Number of output eigenvectors in the first embedding."""
        return self.n_comp_1

    @_n_comp_1.setter
    def _n_comp_1(self, x):
        if not np.isscalar(x):
            ValueError("n_comp_1 must be a scalar.")
        self.n_comp_1 = x

    @property
    def _n_comp_2(self):
        """Number of output eigenvectors in the second embedding."""
        return self.n_comp_2

    @_n_comp_2.setter
    def _n_comp_2(self, x):
        if not np.isscalar(x):
            ValueError("n_comp_1 must be a scalar.")
        self.n_comp_2 = x

    @property
    def _klim(self):
        """Limits for the k-means optimization."""
        return self.klim

    @_klim.setter
    def _klim(self, x):
        x = np.array(x)
        if x.size != 2:
            ValueError("klim must have exactly two elements.")
        if x[0] > x[1]:
            ValueError(
                "First element of klim must be equal to or lower than the second element."
            )
        self.klim = x
