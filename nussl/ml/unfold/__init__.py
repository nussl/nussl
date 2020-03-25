"""
Deep unfolding is a type of architecture where an optimization
process like clustering, non-negative matrix factorization and
other EM style algorithms (anything with update functions) are
unfolded as layers in a neural network. In practice this results
in having the operations available to do on torch Tensors. This
submodule collects implementations that allow one to unfold different
optimization processes in a neural network.
"""

from .gaussian_mixture import GaussianMixtureTorch
