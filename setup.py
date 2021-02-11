#!/usr/bin/env python3

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from os import path
import setuptools


here = path.abspath(path.dirname(__file__))


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsepy",
    version="0.0.1-alpha.1",
    author="Reinder Vos de Wael",
    author_email="reinder.vosdewael@gmail.com",
    description="A toolbox for two step diffusion embedding.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ReinderVosdeWael/TwoStepEmbedding",
    packages=setuptools.find_packages(),
    license="BSD",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    test_require=["pytest"],
    install_requires=["brainspace>=0.1.1", "numpy>=1.16.5", "scikit-learn"],
    project_urls={  # Optional
        "Documentation": "to-be-added",
        "Bug Reports": "https://github.com/ReinderVosdeWael/TwoStepEmbedding/issues",
        "Source": "https://github.com/ReinderVosdeWael/TwoStepEmbedding",
    },
)
