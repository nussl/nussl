
Contributing code
=================

#### NOTE: We are currently not accepting any large submissions as we ramp up to a beta release. 

Aug. 21, 2017: As we prepare to release the beta (soon), we will not be accepting 
large pull requets (i.e., new algorithms, large new features). Feel free to still make them,
but know that they will not be triaged until after the beta. Pull requests for bugs 
will still be handled normally. -e


How to contribute
-----------------

The preferred way to contribute to nussl is to fork the 
[main repository](http://github.com/interactiveaudiolab/nussl/) on
GitHub:

1. Fork the [project repository](http://github.com/interactiveaudiolab/nussl):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/nussl.git
          $ cd nussl 

3. Create a branch to hold your changes:

          $ git checkout -b my-feature

   and start making changes. Never work in the ``master`` branch!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-feature

Finally, go to the web page of the your fork of the nussl repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the committers. The committers will
probably 

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

It is recommended to check that your contribution complies with the
following rules before submitting a pull request:

-  All public methods should have informative docstrings with sample
   usage presented.

-  The documentation follows the [Google Python Style guide](http://www.sphinx-doc.org/en/stable/ext/example_google.html).

-  The coding style complies with PEP8. [Here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01) 
   is a good guide on coding sytle in python.
   
-  If you are contributing an algorithm that is not your own, we ask that you:
   1) Get permission from the algorithm's original author in writing that it is okay
   to include in nussl, and 
   2) Include original or reference code (and an example
   audio file) in the tests directory so that we may benchmark it.
   
-  Write tests for your new code!

You can also check for common programming errors with the following
tools:

<!-- -  Code with good unittest coverage (at least 80%), check with:

          $ pip install nose coverage
          $ nosetests --with-coverage --cover-package=librosa -w tests/ -->

-  No pyflakes warnings, check with:

           $ pip install pyflakes
           $ pyflakes path/to/module.py

-  No PEP8 warnings, check with:

           $ pip install pep8
           $ pep8 path/to/module.py

-  AutoPEP8 can help you fix some of the easy redundant errors:

           $ pip install autopep8
           $ autopep8 path/to/pep8.py

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/interactiveaudiolab/nussl/issues?q=)
   or [pull requests](https://github.com/interactiveaudiolab/nussl/pulls?q=).
   
-  Include all relevant code and/or audio files so that the developers can **reproduce**
   your issue. Reproducible bugs are nussl developers' second favorite things (the first is ice cream).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, scikit-learn, numpy, scipy, librosa, and nussl versions. This information
   can be found by runnning the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import librosa; print("librosa", librosa.__version__)
  import nussl; print("nussl", nussl.__version__)
  ```

Documentation
-------------

Documentation is only in the ``gh-pages`` branch.
After switching to the ``gh-pages`` branch, you can edit the documentation
using any text editor and then generate
the HTML output by typing ``make html`` from the docs/ directory.
The resulting HTML files will be placed in _build/html/ and are viewable 
in a web browser. See [this wiki page](https://github.com/interactiveaudiolab/nussl/wiki/Generating-Documentation) for more information.
Documentation in nussl follows the [Google Python Style guide](http://www.sphinx-doc.org/en/stable/ext/example_google.html).

For building the documentation, you will need 
[sphinx](http://sphinx.pocoo.org/), and
[matplotlib](http://matplotlib.sourceforge.net/).

Note
----
This document was gleefully borrowed from [librosa](https://librosa.github.io/librosa/) (which was gleefully borrowed from [scikit-learn](http://scikit-learn.org/)).
