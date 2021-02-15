""" Tests for the TwoStepEmbedding class. """
import numpy as np
from tsepy import TwoStepEmbedding
from sklearn.utils.estimator_checks import check_estimator

def test_TwoStepEmbedding_output():
    """Tests whether the outputs of TwoStepEmbedding have the correct shape."""
    x = np.random.rand(100, 50, 3)
    y = np.random.rand(100, 50, 3)
    tse = TwoStepEmbedding(
        kernel="cosine",
        approach="dm",
        random_state=0,
        n_comp_1=10,
        n_comp_2=10,
        klim=[2, 10],
    )
    tse.fit(x, out_sample=y)
    assert tse.psi_1.shape == (x.shape[0], tse.n_comp_1, x.shape[2])
    assert tse.psi_2.shape == (x.shape[0] + y.shape[0], tse.n_comp_1)