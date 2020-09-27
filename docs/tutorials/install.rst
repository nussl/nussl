.. _installation_instructions:

Installation
============

There are many ways to install *nussl*. The easiest way is to use the Python Package Index (PyPI),
which is used through the command line with
`pip <https://en.wikipedia.org/wiki/Pip_(package_manager)>`_. pip comes pre-installed with most
python distributions and will automatically install *nussl's* required dependencies.


Requirements
------------

*nussl* is compatible with only **Python 3**.

The requirements are listed in ``requirements.txt`` and should be installed automatically
with `pip`. There is one additional optional requirement that is only needed if you 
want to use ``nussl.separation.primitive.Melodia``: ``vamp``.

If you want to use Melodia, then you will also need to follow the instructions 
`here <https://github.com/justinsalamon/melodia_python_tutorial/blob/master/melodia_python_tutorial.ipynb>`_.

If you are installing from source then you will need to install the requirements manually.

Finally, there are extra requirements you will need to run the examples, tutorials, and 
recipes. These extra requirements are mostly for the sake of visualization. 
Those are in ``extra_requirements.txt``::

      pip install -r extra_requirements.txt

Non-python dependencies
-----------------------

*nussl* has one non-python dependency:
- FFmpeg: https://ffmpeg.org/

If you are installing *nussl* on Windows, you will also need:
- SoX: http://sox.sourceforge.net/

On Linux/macOS SoX is replaced by [SoxBindings](https://github.com/pseeth/soxbindings) which is significantly 
faster. On these platforms SoxBindings is installed automatically 
when calling `pip install nussl` (see below).

On macOS ffmpeg can be installed using `homebrew <https://brew.sh/>`_:

>>> brew install ffmpeg

On linux you can use your distribution's package manager, e.g. on Ubuntu (15.04 "Vivid Vervet" or newer):

>>> sudo apt-get install ffmpeg

NOTE: on earlier versions of Ubuntu `ffmpeg may point to a Libav binary <http://stackoverflow.com/a/9477756/2007700>`_
which is not the correct binary. If you are using anaconda, you can install the correct version by calling:

>>> conda install -c conda-forge ffmpeg

Otherwise, you can `obtain a static binary from the ffmpeg website <https://ffmpeg.org/download.html>`_.

On Windows you can use the provided installation binaries:

- SoX: https://sourceforge.net/projects/sox/files/sox/
- FFmpeg: https://ffmpeg.org/download.html#build-windows

Try to install ``tree`` for the tutorials. This is a useful command line tool 
that shows you the structure of a directory. Install it via a package manager::

      brew install tree # on macosx
      sudo apt-get install tree # on ubuntu

pip install
-----------

Installing *nussl* is easy using pip and the Python Package Index. This will install all required dependencies alongside
the *nussl* installation. Use the following command::

        pip install nussl

to install.

Running the tests
-----------------

To run the tests, clone the repository, then `cd` into the directory and do::

        pip install -r requirements.txt
        pip install -r extra_requirements.txt
        # install vamp and melodia as above
        pytest

Installing from source and downloading
--------------------------------------

Alternatively, you may download the source code from `github <https://github.com/interactiveaudiolab/nussl>`_
and install with the following command::

        python setup.py nussl


It is also possible to download from github and copy the *nussl* folder into your working directory and
directly import the modules in your python code.


.. _troubleshooting:

Troubleshooting
---------------

Matplotlib issues
^^^^^^^^^^^^^^^^^

If you get this issue:

>>> import matplotlib.pyplot as plt
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "//anaconda/lib/python2.7/site-packages/matplotlib-1.3.1-py2.7-macosx-10.5-x86_64.egg/matplotlib/pyplot.py", line 98, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "//anaconda/lib/python2.7/site-packages/matplotlib-1.3.1-py2.7-macosx-10.5-x86_64.egg/matplotlib/backends/__init__.py", line 28, in pylab_setup
    globals(),locals(),[backend_name],0)
  File "//anaconda/lib/python2.7/site-packages/matplotlib-1.3.1-py2.7-macosx-10.5-x86_64.egg/matplotlib/backends/backend_macosx.py", line 21, in <module>
    from matplotlib.backends import _macosx
**RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends.

Then there is an issue with your `matplotlib backend renderer <http://matplotlib.org/1.3.0/faq/usage_faq.html#what-is-a-backend>`_.
This is an easy fix. There is a `stack overflow post here <http://stackoverflow.com/q/21784641/5768001>`_ about it.

The solution is to create a matplotlib config file here: ``~/.matplotlib/matplotlibrc`` and add this line to it:
``backend: TkAgg``.

You might get some warnings now when you ``import nussl`` but this is fine. You can also switch to another backend.
To see what backends are available, run this code:

>>> import matplotlib.pyplot as plt
>>> fig = plt.figure()
>>> print(fig.canvas.get_supported_filetypes())

Contact
^^^^^^^

Please get in contact or open a `github issue <https://github.com/interactiveaudiolab/nussl/issues>`_ if you encounter any installation issues.


