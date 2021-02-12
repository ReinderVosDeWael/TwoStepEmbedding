import pytest
import numpy as np
from tsepy.base import TwoStepEmbedding


def test_TwoStepEmbedding():
    x = np.random.rand(100, 50, 3)
    tse = TwoStepEmbedding(
        kernel="cosine", approach="dm", rng=0, d1=10, d2=10, klim=[2, 10]
    )
    tse.fit(x)
    assert tse.psi_1.shape == (x.shape[0], tse.d1, x.shape[2])
    assert tse.psi_2.shape == (x.shape[0], tse.d2)
