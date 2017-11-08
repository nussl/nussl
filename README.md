
nussl
=====

**The Northwestern University Source Separation Library (nussl)** (pronounced ["nuzzle"](http://www.thefreedictionary.com/nuzzle)) is a flexible, object oriented python 
audio source separation library created by the [Interactive Audio Lab](http://music.cs.northwestern.edu/) 
at Northwestern University. At its core, nussl provides implementations of common source separation
algorithms as well as an easy-to-use framework for prototyping and adding new algorithms. The aim of nussl is
to create a low barrier to entry for using popular source separation algorithms, while also allowing the
user fine tuned control of low-level parameters.


**NOTICE: nussl is currently in alpha. Please be mindful.**

Please see the issues page before contacting the authors.

Branch Layout
-------------

- **Master** contains the most recent stable version, the version that you get when you pip install *nussl*.
- **Development** contains the most recent work, but is not as stable as master. Most all of development work happens
in this branch before being pushed to **master**.
- **Experimental** has many more implementations, but many have not been vetted or properly tested. Some methods in this
branch require extra functionality that cannot be included in a pip install, such as the Vamp binary files or 
tensorflow. This branch is the bleeding edge.
- **gh-pages** is used to auto generate our documentation using Sphinx.
- Other feature branches do exist from time to time as well.



Documentation
-------------

Full documentation is [available here.](https://interactiveaudiolab.github.io/nussl/)

Note: This package has been tested with python 2.7, but not python 3.x yet. Use python 3 at your own peril!


Features
--------

At its core, nussl contains implementations of the following source separation algorithms:

Spatialization algorithms:
* Degenerate Unmixing Estimation Technique (DUET)
* PROJET 

Median filtering algorithms:
* REpeating Pattern Extraction Technique (REPET)
* REPET using the cosine similarity matrix (REPET-SIM)
* Harmonic/Percussive Source Separation (HPSS)
* Kernel Adaptive Modeling (KAM) 

General matrix decomposition:
* Non-negative Matrix Factorization (NMF)
* Robust Principal Component Analysis (RPCA) 
* Independent Component Analysis (ICA)

Other Separation Methods
* Ideal Mask
* Overlap Add
* Algorithm Picker (multicue separation)
* Separation via 2DFT
* Melody Tracking separation (Melodia)

Also coming: Deep methods (Deep Clustering, Huang et. al. Deep Separation, etc.) (with keras backend), Separation by 
Segmentation, Melody tracking methods (Melodia, SIMM), and many more! Note: newly algorithms usually do not live in the master branch. 

The nussl also has an evaluation framework, which provides interoperability
between nussl and [mir_eval](https://github.com/caffel/mir_eval) (a python implementation of [BSSEval](http://bass-db.gforge.inria.fr/bss_eval)) as well as implementations of other evaluation methods. 


Installation
------------

We recommend getting installing anaconda prior to using this package.

Installation is easy if you have pip (recommended):
```
$ pip install nussl
```

A note on cloning from github: **nussl** contains a large number of uncompressed audio benchmark files for testing which makes cloning directly from Github a slow process. **nussl** is now synced with [Git LFS](https://git-lfs.github.com), so if you want to clone from Github without downloading the large testing files, please use Git LFS.


Development Status
------------------

Camera ready (in master and pip builds):
* Repet
* RepetSim
* IdealMask
* OverlapAdd
* DUET 


In Development Branch:
* NMF
* RPCA
* PROJET
* 2DFT modulation separation

Planned:
* KAM
* HPSS
* SIMM
* Separation via segmentation
* Deep clustering
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
