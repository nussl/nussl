Changelog
=========

v1.1.9
------
- Adding option to clip the minimum value of SI-SDR loss.

v1.1.8
------
- Exposing window type in OverlapAdd.
- Updated torch.fft -> torch.fft.rfft so it works with PyTorch 1.8.
- Added an option to modules in SeparationModel where the forward pass is
  made simpler.

v1.1.7
------
- Added load function to SeparationModel.
- Allow keyword arguments to pass transparently through a SeparationModel.
- Added runtime arguments to DeepMixin, and a callback that can be implemented by a user
  to modify an input data dictionary before it's passed to a SeparationModel.
- Added an OverlapAdd algorithm that can be used with any separation object.
- Adding argument to MelProjection so that it matches librosa mel filters.

v1.1.5
------
- This version was skipped because of a failed PyPI release.

v1.1.4
------
- Fixed visualization of mel spectrogram.

v1.1.3
------
- Fixed some bugs that happen because of a PyTorch Ignite update.
- Fixed a bug in effects where the sample rate wouldn't propagate through some effects.
- Added more metadata that can be saved along with a model, using an associate_metrics function.
- Updated handlers for printing after each epoch during training and writing to tensorboard.
- Added interaction via gradio.

v1.1.2
------
- Fixed a bug in Dual Path models where they couldn't be used with DataParallel.
- Fixed a regularization bug in SI-SDR (eps=1e-8 instead of 1e-10).

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
