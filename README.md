
nussl
=====

This branch is an update/refactor of ``nussl`` to update to Python 3 and have complete code coverage via the tests.

Current progress
----------------
A check below doesn't happen until full test coverage is achieved.

- [x] Core AudioSignal functionality
- [x] STFT/iSTFT via ``scipy.signal``
- [x] External file zoo functionality
- [x] Hooks for common datasets
    - [x] MUSDB
    - [x] Scaper
    - [x] MixSourceFolder 
    - [ ] WHAM! (simple subclass MixSourceFolder)
    - [ ] MIR1K
    - [ ] iKala (still support?)
  - [x] Evaluation  
    - [x] SISDR
    - [x] museval BSSEval
    - [x] Precision/recall/f1-score on masks
  - [ ] Machine learning
    - [x] Network modules
    - [x] SeparationModel
    - [x] PyTorch Gaussian Mixture model
    - [ ] Trainer via PyTorch ignite
  - [ ] Separation (will fill this in more soon)
    - [ ] Deep methods
    - [ ] Existing methods

Current test coverage
---------------------

```
---------- coverage: platform darwin, python 3.7.6-final-0 -----------
Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
nussl/core/__init__.py                            5      0   100%
nussl/core/audio_signal.py                      431      0   100%
nussl/core/constants.py                          38      0   100%
nussl/core/efz_utils.py                         145      0   100%
nussl/core/jupyter_utils.py                      33      0   100%
nussl/core/masks/__init__.py                      4      0   100%
nussl/core/masks/binary_mask.py                  23      0   100%
nussl/core/masks/mask_base.py                    92      0   100%
nussl/core/masks/soft_mask.py                    16      0   100%
nussl/core/utils.py                             108      0   100%
nussl/datasets/__init__.py                        2      0   100%
nussl/datasets/base_dataset.py                   32      0   100%
nussl/datasets/hooks.py                          68      0   100%
nussl/datasets/transforms.py                    149      0   100%
nussl/evaluation/__init__.py                      4      0   100%
nussl/evaluation/bss_eval.py                     63      0   100%
nussl/evaluation/evaluation_base.py              64      0   100%
nussl/evaluation/precision_recall_fscore.py      34      0   100%
nussl/ml/networks/__init__.py                     1      0   100%
nussl/ml/networks/builders.py                    18      0   100%
nussl/ml/networks/modules.py                    188      0   100%
nussl/ml/networks/separation_model.py            78      0   100%
nussl/ml/unfold/__init__.py                       1      0   100%
nussl/ml/unfold/gaussian_mixture.py              74      0   100%
---------------------------------------------------------------------------
TOTAL                                          1671      0   100%

=============================================================================== 230 passed, 9 warnings in 201.19s (0:03:21) 
```


**The Northwestern University Source Separation Library (nussl)**
(pronounced ["nuzzle"](http://www.thefreedictionary.com/nuzzle)) is a flexible, object oriented
python audio source separation library created by the 
[Interactive Audio Lab](http://music.cs.northwestern.edu/) 
at Northwestern University. At its core, nussl provides implementations of common source separation
algorithms as well as an easy-to-use framework for prototyping and adding new algorithms. The aim of
nussl is to create a low barrier to entry for using popular source separation algorithms, while also
allowing the user fine tuned control of low-level parameters.


**NOTICE: nussl is currently in alpha. Please be mindful.**

Please see the issues page before contacting the authors.

Branch Layout
-------------

- **Master** contains the most recent stable version, the version that you get when you pip install 
*nussl*.
- **Development** contains the most recent work, but is not as stable as master. Most all of 
development work happens in this branch before being propagated to **master**.
- **Experimental** has many more implementations, but many have not been vetted or properly tested. 
Some methods in this branch require extra functionality that cannot be included in a pip install, 
such as the Vamp binary files or pytorch.
- **gh-pages** is used to auto generate our documentation using Sphinx.
- Other feature branches do exist from time to time as well.



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