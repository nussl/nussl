
nussl
=====

**nussl** (pronounced ["nuzzle"](http://www.thefreedictionary.com/nuzzle)) is a flexible, object oriented python 
audio source separation library created by the [Interactive Audio Lab](http://music.cs.northwestern.edu/) 
at Northwestern University. At its core, nussl provides implementations of common source separation
algorithms as well as an easy-to-use framework for prototyping and adding new algorithms. The aim of nussl is
to create a low barrier to entry for using popular source separation algorithms, while also allowing the
user fine tuned control of low-level parameters.


**NOTICE: nussl is currently in alpha. Please be mindful.**

Please see the issues page before contacting the authors.

Documentation
-------------

Full documentation is [available here.](https://interactiveaudiolab.github.io/nussl/)

Note: This package has been tested with python 2.7, but not python 3.x yet. Use python 3 at your own peril!


Features
--------

At its core, nussl contains (or will contain) implementations of the following source separation algorithms:

Spatialization algorithms:
* Degenerate Unmixing Estimation Technique (DUET)
* PROJET 

Median filtering algorithms:
* REpeating Pattern Extraction Technique (REPET)
* REPET using the cosine similarity matrix (REPET-SIM)
* Harmonic/Percussive Source Separation (HPSS) (**Coming Soon**)
* Kernel Adaptive Modeling (KAM) (**Coming Soon**)

General matrix decomposition:
* Non-negative Matrix Factorization (NMF) (**Coming Soon**)
* Robust Principal Component Analysis (RPCA) (**Coming Soon**)

Also coming: Deep methods (Deep Clustering, Huang et. al. Deep Separation, etc.) (with keras backend), Separation by Segmentation, Melody tracking methods (Melodia, SIMM), ICA, Ideal Masks, Overlap adding (windowing) functionality for existing methods, and many more!

The nussl team is also working on implementing an evaluation framework, which will include easy interoperability
between nussl and `mir_eval` (`bss_eval`) as well as implementations of other evaluation methods. (See development
branch)


Installation
------------

We recommend getting installing anaconda prior to using this package.

Installation is easy if you have pip (recommended):
```
$ pip install nussl
```

--or--

Once you've downloaded and decompressed the tarball, run setup.py like this:
```
$ python setup.py install
```

You can also download and copy the nussl folder into your working directory.


nussl requires:

```
numpy        version >= 1.8.0
scipy        version >= 0.13.0
matplotlib   version >= 1.3.1
audioread    version >= 2.1.2
librosa      version >= 0.4.1
```


Development Status
------------------

Camera ready (in master and pip builds):
* REPET
* REPET-SIM

In Development Branch:
* DUET (Just touch-up work)
* NMF
* RPCA
* PROJET
* 2DFT modulation separation

Planned:
* KAM
* HPSS
* SIMM
* Separation via segmentation
* Your algorithm? (contact us for details)


License
-------
nussl 0.1.5* is under an [MIT License](https://opensource.org/licenses/MIT)

The MIT License (MIT)

Copyright (c) 2017 Interactive Audio Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Contributors
------------
Ethan Manilow ([website](http://www.ethanmanilow.com)),
Prem Seetharaman ([website](http://prem.seeth.org/)),
Fatemah Pishdadian ([website](http://fatemehpishdadian.com/))

Former:

Corey Grief ([website](http://music.cs.northwestern.edu/emeritus.php)),


Contact
-------
Contact Ethan Manilow (ethanmanilow [at] u.northwestern.edu) with any questions or issues. Please look at the
"issues" page before reporting problems.
