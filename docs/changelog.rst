Changelog
=========

v1.1.2
------
- Fixed a bug in Dual Path models where they couldn't be used with DataParallel.

v1.1.1
------
- Allowed MixSourceFolder to mix sources on the fly.
- Adding OnTheFly dataset for mixing sources on the fly using a function.
- Added effects utilities for data augmentation.
- Fixed bug with ignite 0.4.0

v1.1.0
------
- Added building blocks for dual path recurrent neural networks
- Added a builder for end-to-end networks and for dual path recurrent networks
- Added confidence measures that can predict separation performance for clustering-based
  algorithms
- Improved tests and documentation
- Fixed a minor bug in GetExcerpt
- Made gpytorch a dependency for faster differentiable GMM

v1.0.2
------
- Fixed an issue with backprop in SI-SDR loss.
- Added a "verbose" argument to SeparationModel to make it easier
  to debug.

v1.0.1
------
- Added long description to PyPi. Released onto PyPi officially!

v1.0.0
------

- Python 3 support!
- Much improved documentation and API
- Refactoring of AudioSignal and other parts to be more intuitive
- Full test coverage
- Deep learning algorithms added, along with robust model zoo
- PyTorch fully integrated into nussl
- Support for many kinds of datasets, and hooks for existing ones
  like MUSDB, WHAM!, and FUSS.
- Easy-to-use framework for training new separation models.
- Improved performance of many of the classical non-ML based
  separation algorithms.

v0.1.6a0
--------
Initial public release.

Highlights:

- Presentation at ISMIR 2018
- Adding EFZ utilities
- BSS-Eval v4 support
