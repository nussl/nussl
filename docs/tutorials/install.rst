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
want to use `nussl.separation.primitive.Melodia`: `vamp`.

If you want to use Melodia, then you will also need to follow the instructions 
`here <https://github.com/justinsalamon/melodia_python_tutorial/blob/master/melodia_python_tutorial.ipynb>`_

If you are installing from source then you will need to install the requirements manually.

Finally, there are extra requirements you will need to run the recipes. Those are in
``extra_requirements.txt``.

.. _anaconda_env:

pip install
-----------

Installing *nussl* is easy using pip and the Python Package Index. This will install all required dependencies alongside
the *nussl* installation. Use the following command::

        pip install nussl

to install.

Running the tests
-----------------

To run the tests, clone the repository, then `cd` into the directory and do::

        pip install -r requirements
        pip install -r extra_requirements
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


