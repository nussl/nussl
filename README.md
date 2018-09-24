
nussl
=====

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

Note: This package has been tested with python 2.7, but not python 3.x yet. python3 coming soon!


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

Contributors
------------
Ethan Manilow ([website](http://www.ethanmanilow.com)),
Prem Seetharaman ([website](http://prem.seeth.org/)),
Fatemeh Pishdadian ([website](http://fatemehpishdadian.com/))

Former:

Corey Grief ([website](http://music.cs.northwestern.edu/emeritus.php)), 
Daniel Felix Kim, Ben Kalish


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