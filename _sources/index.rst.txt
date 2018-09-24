.. nussl documentation master file, created by
   sphinx-quickstart on Wed Jan 13 00:16:58 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nussl
=====
*nussl* (pronounced `nuzzle <http://www.thefreedictionary.com/nuzzle>`_) [#f1]_ is a flexible,
object oriented python audio source separation library created by the
`Interactive Audio Lab <http://music.cs.northwestern.edu>`_ at Northwestern University.
At its core, *nussl* provides implementations of common source separation algorithms as well as an
easy-to-use framework for prototyping and adding new algorithms. The aim of *nussl* is to create a
low barrier to entry for using popular source separation algorithms, while also allowing the user
fine tuned control of low-level parameters.

First Steps
-----------
.. toctree::
    :maxdepth: 2

    Getting Started </getting_started/getting_started>

nussl Reference
---------------
.. toctree::
   :maxdepth: 2

   Core </src/core/core>
   Separation </src/separation/separation_classes>
   Evaluation </src/evaluation/evaluation_classes>
   Transformers </src/transformers/transformer_classes>


Demos
-----
.. toctree::
    :maxdepth: 2

    Code Examples </examples/examples>

`ISMIR 2018 Demo <https://interactiveaudiolab.github.io/demos/nussl.html>`_


Contributing
------------
.. toctree::
    :maxdepth: 2

    Contribution Guide <contributing>


Troubleshooting
---------------
For bug reports and issues with this code, please see the `github issues page
<https://github.com/interactiveaudiolab/nussl/issues>`_. Please review open issues before contacting
the authors.

Changelog
---------
.. toctree::
    :maxdepth: 1

    Changelog <changelog>

Contact
-------
Contact Ethan Manilow <ethanm[at]u.northwestern[dot]edu> with any questions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: Footnotes

.. [#f1] Officially stands for "Northwestern University Source Separation Library", but in our
         hearts *nussl* stands for "Needs Unmixing? Source Separation Library!"
