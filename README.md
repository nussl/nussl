nussl
=====

This branch is an update/refactor of *nussl* to update to Python 3 and have complete 
code coverage via the tests.

Current test coverage
---------------------

Clone this repo, then run

```
pip install -r requirements.txt
pip install -r extra_requirements.txt
pytest
```

```
---------- coverage: platform darwin, python 3.7.6-final-0 -----------
Name                                                  Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------------
nussl/__init__.py                                        11      0   100%
nussl/core/__init__.py                                    6      0   100%
nussl/core/audio_signal.py                              453      0   100%
nussl/core/constants.py                                  39      0   100%
nussl/core/efz_utils.py                                 145      0   100%
nussl/core/masks/__init__.py                              4      0   100%
nussl/core/masks/binary_mask.py                          23      0   100%
nussl/core/masks/mask_base.py                            92      0   100%
nussl/core/masks/soft_mask.py                            16      0   100%
nussl/core/mixing.py                                     32      0   100%
nussl/core/play_utils.py                                 37      0   100%
nussl/core/utils.py                                     138      0   100%
nussl/datasets/__init__.py                                2      0   100%
nussl/datasets/base_dataset.py                           80      0   100%
nussl/datasets/hooks.py                                  87      0   100%
nussl/datasets/transforms.py                            271      0   100%
nussl/evaluation/__init__.py                              3      0   100%
nussl/evaluation/bss_eval.py                             60      0   100%
nussl/evaluation/evaluation_base.py                      84      0   100%
nussl/evaluation/precision_recall_fscore.py              34      0   100%
nussl/ml/__init__.py                                     10      0   100%
nussl/ml/cluster.py                                       2      0   100%
nussl/ml/networks/__init__.py                             2      0   100%
nussl/ml/networks/builders.py                            18      0   100%
nussl/ml/networks/modules.py                            182      0   100%
nussl/ml/networks/separation_model.py                    74      0   100%
nussl/ml/train/__init__.py                                3      0   100%
nussl/ml/train/closures.py                               78      0   100%
nussl/ml/train/loss.py                                   98      0   100%
nussl/ml/train/trainer.py                               122      0   100%
nussl/ml/unfold/__init__.py                               1      0   100%
nussl/ml/unfold/gaussian_mixture.py                      74      0   100%
nussl/separation/__init__.py                              2      0   100%
nussl/separation/base/__init__.py                         5      0   100%
nussl/separation/base/clustering_separation_base.py      56      0   100%
nussl/separation/base/deep_mixin.py                      42      0   100%
nussl/separation/base/mask_separation_base.py            56      0   100%
nussl/separation/base/nmf_mixin.py                       35      0   100%
nussl/separation/base/separation_base.py                 50      0   100%
nussl/separation/benchmark/__init__.py                    4      0   100%
nussl/separation/benchmark/high_low_pass_filter.py       13      0   100%
nussl/separation/benchmark/ideal_binary_mask.py          21      0   100%
nussl/separation/benchmark/ideal_ratio_mask.py           27      0   100%
nussl/separation/benchmark/wiener_filter.py              25      0   100%
nussl/separation/composite/__init__.py                    1      0   100%
nussl/separation/composite/ensemble_clustering.py        71      0   100%
nussl/separation/deep/__init__.py                         2      0   100%
nussl/separation/deep/deep_clustering.py                 20      0   100%
nussl/separation/deep/deep_mask_estimation.py            33      0   100%
nussl/separation/factorization/__init__.py                2      0   100%
nussl/separation/factorization/ica.py                    39      0   100%
nussl/separation/factorization/rpca.py                   73      0   100%
nussl/separation/primitive/__init__.py                    6      0   100%
nussl/separation/primitive/ft2d.py                      109      0   100%
nussl/separation/primitive/hpss.py                       25      0   100%
nussl/separation/primitive/melodia.py                   123      0   100%
nussl/separation/primitive/repet.py                     101      0   100%
nussl/separation/primitive/repet_sim.py                  78      0   100%
nussl/separation/primitive/timbre.py                     24      0   100%
nussl/separation/spatial/__init__.py                      3      0   100%
nussl/separation/spatial/duet.py                        106      0   100%
nussl/separation/spatial/projet.py                      132      0   100%
nussl/separation/spatial/spatial_clustering.py            9      0   100%
-----------------------------------------------------------------------------------
TOTAL                                                  3574      0   100%

============================================================================== 311 passed, 21 warnings in 1047.44s (0:17:27) 
```


**The Northwestern University Source Separation Library (nussl)**
(pronounced ["nuzzle"](http://www.thefreedictionary.com/nuzzle)) is a flexible, object oriented
python audio source separation library created by the 
[Interactive Audio Lab](http://music.cs.northwestern.edu/) 
at Northwestern University. At its core, nussl provides implementations of common source separation
algorithms as well as an easy-to-use framework for prototyping and adding new algorithms. The aim of
nussl is to create a low barrier to entry for using popular source separation algorithms, while also
allowing the user fine tuned control of low-level parameters.



Please see the issues page before contacting the authors.


Documentation
-------------

Full documentation is [available here.](https://interactiveaudiolab.github.io/nussl/)


Features
--------

At its core, nussl contains implementations of the following source separation algorithms:

Spatialization algorithms:
* Degenerate Unmixing Estimation Technique (DUET)
* PROJET 

Repetition algorithms:
* REpeating Pattern Extraction Technique (REPET)
* REPET using the cosine similarity matrix (REPET-SIM)
* Separation via 2DFT

General matrix decomposition/Component Analysis:
* Non-negative Matrix Factorization with MFCC clustering (NMF)
* Robust Principal Component Analysis (RPCA) 
* Independent Component Analysis (ICA)

Benchmarking
* Ideal Mask
* High/Low Pass Filtering

Composite Methods
* Overlap Add
* Algorithm Picker (multicue separation)

Other Foreground/Background Decompositions
* Harmonic/Percussive Source Separation (HPSS)
* Melody Tracking separation (Melodia)

Deep Learning
* Deep Clustering

*Your Algorithm Here*

The nussl also has an evaluation framework, which provides interoperability
between nussl and [mir_eval](https://github.com/craffel/mir_eval) (a python implementation of
 [BSSEval](http://bass-db.gforge.inria.fr/bss_eval)) as well as implementations of other 
 evaluation methods. 


Installation
------------

We recommend getting installing anaconda prior to using this package.

Installation is easy if you have pip (recommended):

```
$ pip install nussl
```

Or if you want to install from source (some sort of Python environment is recommended):

```
$ git clone https://github.com/interactiveaudiolab/nussl.git
$ cd nussl
$ pip install -r requirements.txt
# or
$ pip install -e .
```

Citing
------

If you are using nussl for your research project, we please ask that you cite it using one of the 
following bibtex citations:

    @inproceedings {nussl
        author = {Ethan Manilow and Prem Seetharaman and Bryan Pardo},
        title = "The Northwestern University Source Separation Library",
        publisher = "Proceedings of the 19th International Society of Music Information Retrieval 
            Conference ({ISMIR} 2018), Paris, France, September 23-27",
        year = 2018
    }

    @Misc{nussl,
        author =       {Ethan Manilow and Prem Seetharaman and Fatemeh Pishdadian and Bryan Pardo},
        title =        {{NUSSL}: the Northwestern University Source Separation Library},
        howpublished = {\url{https://github.com/interactiveaudiolab/nussl}},
        year =        {2018}
    }

Contributions
-------------

See the [contribution guide](https://interactiveaudiolab.github.io/nussl/contributing.html) for
detailed information. But the basics are: bug fixes/enhancements/etc have the standard github
process; but, when adding new algorithms, contributors must provide benchmark files, paper 
references, and trained models (if applicable).

## Contributors

- Ethan Manilow <http://github.com/ethman>
- Prem Seetharaman <http://github.com/pseeth>
- Fatemeh Pishdadian <http://github.com/fpishdadian>
- Nathan Shelly

Former:

- Corey Grief <http://github.com/cegrief>
- Daniel Felix Kim <http://github.com/DanielFelixKim>
- Ben Kalish

nussl's REPET, REPET-SIM implementations are based on MATLAB code from Zafar Raffi. 
The DUET implementation is based on MATLAB code from Scott Richard. 
nussl's PROJET implementation is based on python code from Antoine Liutkus.
Similarly for most of the algorithms in nussl. 

See documentation and inline comments for each algorithm for more information about citations and authorship.


Contact
-------
Contact Ethan Manilow (ethanmanilow [at] u [dot] northwestern [dot] edu) with any questions or 
issues. Please look at the "issues" page before reporting problems.


License
-------
nussl 0.1.6* is under an [MIT License](https://opensource.org/licenses/MIT)

The MIT License (MIT)

Copyright (c) 2018 Interactive Audio Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
