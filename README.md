# The Northwestern University Source Separation Library


**The Northwestern University Source Separation Library (nussl)**
(pronounced ["nuzzle"](http://www.thefreedictionary.com/nuzzle)) is a flexible, object oriented
python audio source separation library created by the 
[Interactive Audio Lab](http://music.cs.northwestern.edu/) 
at Northwestern University. Whether you're a researcher or just need an out-of-the-box source 
separation model, nussl contains everything you need for modern source separation.
nussl is to create a low barrier to entry for using popular source separation algorithms, while also
allowing the user fine tuned control of low-level parameters.


# Documentation

Tutorials, code examples, and the API reference are available at the documentation
website which is [available here.](https://interactiveaudiolab.github.io/nussl/)


# Features


## Deep Learning


nussl contains a rich feature set for deep learning models, including a 
built in trainer with support for many common datasets in music and speech separation.
nussl also has a External File Zoo where users are able to download pre-trained
models on many different datasets with a number of different configurations. It is also 

### Training Your Own Algorithms

nussl makes it easy to train your own algorithms, with many common architectures,
datasets, and configurations ready-built for many use cases, and an easy way to
extend our built-in functionality for your own needs. There are many pre-built useful
modules that are easy to piece together, such as LSTMs, GRUs, convolutional layers,
FC layers, mask layers, embedding layers. Everything is built on Pytorch, so adding
a brand new module is seemless.

### Using Your Own Data

nussl has existing support for many popular source separation datasets, like MUSDB18,
WSJ0, WHAM! nussl also defines a simple directory structure to add in any data you want.
Want to augment your data? We have support for Scaper output, built in so you
can train on magnitudes more training data. 

### Downloading Pre-trained Models

Pre-trained models on speech separation, music separation, and more are available
for download at the [External File Zoo (EFZ)](http://nussl.ci.northwestern.edu/),
and there is a build in python API to download models you want from within your code.

## Deep Learning Architectures

We provide the following architectures available out of the box:
* Mask Inference
* Deep Clustering
* Chimera
* OpenUnmix
* TasNet
* DualPath RNN

(See docs for corresponding citations)

## Classical Algorithms

Additionally, nussl also contains implementations of many classical source separation
algorithms, including:

* Spatialization algorithms (DUET, PROJET)
* Factorization-based Methods (RPCA, ICA)
* Primitive Methods (Repetition [REPET, 2DFT], Harmonic/Percussive, Vocal melody extraction, Timbre clustering)
* Benchmark Methods (High Pass filter, Binary Masks, Ratio Masks, Weiner Filtering)


## Evaluation

nussl has many built in evaluation measures to determine how well your algorithm did, including
the museval measure (BSSEvalV4), SI-SDR, and others.



# Installation


nussl is on the Python Package Index (PyPI), so it is easy to install with pip:

```
$ pip install nussl
```

**Note: Requires python 3.7+**


# Citing

If you are using nussl for your research project, we please ask that you cite it using one of the 
following bibtex citations:

    @inproceedings {nussl
        author = {Ethan Manilow and Prem Seetharaman and Bryan Pardo},
        title = "The Northwestern University Source Separation Library",
        publisher = "Proceedings of the 19th International Society of Music Information Retrieval 
            Conference ({ISMIR} 2018), Paris, France, September 23-27",
        year = 2018
    }


# Issues and Contributions

See the [contribution guide](https://interactiveaudiolab.github.io/nussl/contributing.html) for
detailed information. But the basics are: bug fixes/enhancements/etc have the standard github
process; but, when adding new algorithms, contributors must provide benchmark files, paper 
references, and trained models (if applicable). Please see the issues page before contacting the authors.

# Contributors

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


# Contact

Contact Ethan Manilow (ethanm [at] u [dot] northwestern [dot] edu) or 
Prem Seetharaman (prem [at] u [dot] northwestern [dot] edu) with any questions or 
issues. Please look at the "issues" page before reporting problems.


# License

nussl 1.0 is under an [MIT License](https://opensource.org/licenses/MIT)

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
