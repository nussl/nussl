.. _installation_instructions:

Installation
============
**It is highly recommended to install** *nussl* **inside a**
`venv <https://docs.python.org/3/library/venv.html>`_ **because it is under active development.**
We also recommend installing `anaconda <https://www.continuum.io/>`_ prior to using this package.

pip install
-----------

Installing *nussl* is easy using pip and the Python Package Index. This will install all required dependencies alongside the
*nussl* installation. Use the following command::

        pip install nussl

or::

        sudo pip install nussl

to install.

Installing from source and downloading
--------------------------------------

Alternatively, you may download the source code from `github <https://github.com/interactiveaudiolab/nussl>`_
and install with the following command::

        python setup.py nussl



It is also possible to download from github and copy the *nussl* folder into your working directory and
directly import the modules in your python code.

Requirements
------------

*nussl* requires:

* ``python 2.7``
* ``numpy version >= 1.8.0``
* ``scipy version >= 0.13.0``
* ``matplotlib version >= 1.3.1``
* ``audioread >= 2.1.2``
* ``librosa >= 0.4.1``

The first four are satisfied by `anaconda <https://www.continuum.io/>`_.


.. _troubleshooting:

Troubleshooting
---------------

Making sure you have the correct version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes issues arise that can be fixed by making sure you version of *nussl* is up to date. To check if you have the
most up-to-date version of nussl, check the version number here: https://pypi.python.org/pypi/nussl and compare it with
the one you are accessing in your python like so:

>>> import nussl
>>> nussl.version
0.1.5a9

If the version numbers match then you're set. If not you can get the most recent version in a few ways. From the
terminal, the command::

        pip install -U nussl

will force pip to install the newest version of nussl and all of its dependencies.

In the case that this fails, you can force pip to install a specific version like this::

        pip install nussl==[version]

So if I find that version 0.1.5a9 is the most recent (by checking PyPI), my command would look like the following::

        pip install nussl==0.1.5a9


pip issues with anaconda
^^^^^^^^^^^^^^^^^^^^^^^^

If you have anaconda installed on your machine and after a ``pip`` install doing ``import nussl`` is crashing, it might
be the case that pip is installing *nussl* to a (non-anaconda) python binary elsewhere on your machine.
You can target the directory where pip installs *nussl* by adding this flag to your pip command:
``--target=d:\somewhere\other\than\the\default``. See: http://stackoverflow.com/q/2915471/5768001
If you find your anaconda (or anaconda2) folder on your machine, the directory to target should be
``anaconda2/lib/python2.7/site-packages``.

The more inelegant solution is to find your pre-existing *nussl*
installation, which means locating two directories named ``nussl`` and ``nussl-[version].dist-info`` (where [version]
is the version string as above). These are usually in a directory structure like ``lib/python2.7/site-packages``
as above, but not anaconda related, presumably where pip installed them. Once you find those two folders copy and paste
them to the anaconda directory above.


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

Please get in contact or open a github issue if you encounter any installation issues.


