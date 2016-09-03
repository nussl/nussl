=====
nussl
=====

**nussl** (pronounced like "nuzzle") is a flexible python source separation library
created by the [Interactive Audio Lab] (http://music.cs.northwestern.edu/) at Northwestern University.


Features
--------

At its core, nussl contains implementations of the following source separation algorithms:
* DUET
* REPET
* NMF
* KAM
* RPCA

It provides an easy framework for adding new algorithms to experiment with and has some basic functionality for
interacting with audio files in both the time and frequency domains. At the moment, nussl only works with .wav files.
Please see the issues page before contacting the authors.

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


nussl requires: `numpy, version >= 1.8.0`, `scipy, version >= 0.13.0`, `matplotlib, version >= 1.3.1`


Documentation
-------------

Full documentation is [available here.](https://interactiveaudiolab.github.io/nussl/)

Note: This package has been tested with python 2.7, but not 3.\*. Use 3.\* at your own peril!

Status
------
KAM, RPCA (and corresponding demos) need to be migrated to the framework. They will be
more difficult to get running.


License
-------
nussl 0.1.5* is under an [MIT License](https://opensource.org/licenses/MIT)

The MIT License (MIT)

Copyright (c) 2016 Interactive Audio Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Contributors
------------
Corey Grief ([website](http://music.cs.northwestern.edu/people.php)),
Ethan Manilow ([website](http://www.ethanmanilow.com)),
Fatemah Pishdadian ([website](http://music.cs.northwestern.edu/people.php)),

Contact
-------
Contact Ethan Manilow (ethanmanilow2015 [at] u.northwestern.edu) with any questions or issues. Please look at the
"issues" page before reporting problems.