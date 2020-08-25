Contribution Guide
==================

General
-------

Contributions are welcome and encouraged for *nussl*. It is through a community effort that *nussl*
has become what it is. With that in mind, we would like to outline a few things about contributing
to *nussl* before you submit your next amazing pull request.

Contributing to *nussl* is just like contributing to any other open source project, for the
most part. The process is exactly the same for fixing bugs or adding utility functions: make
a pull request, write some tests, make sure all of the code is up to snuff, and it'll be approved.

Documentation
-------------

Developing
^^^^^^^^^^

Documentation for *nussl* is built with Sphinx. The source code for making docs
is kept in the ``docs/`` folder. There are two parts to the documentation. The first
is API documentation which is maintained alongside the code. The second part is
tutorials and examples, which are maintained as Jupyter *notebooks* and compiled into 
the docs via [nbsphinx](https://nbsphinx.readthedocs.io/en/0.5.1/).

The installation requirements for maintaining docs are kept in ``extra_requirements``::

   $ pip install -r extra_requirements.txt

Also make sure that nussl itself is installed in your env by navigating to the top-level
*nussl* directory and running::

   $ pip install -e .

You will also need both of the following installed to edit or create new tutorials or examples:

- [Jupyter](https://jupyter.org/install)
- [Pandoc](https://pandoc.org/installing.html)

Follow the links for installation instructions for each of these projects.
(You may need to pip install extra_requirements.txt outside a conda env so that the
jupyter extensions can register correctly.)

Notebooks are NOT kept in the docs repository as is, as notebooks are very
hard to keep track of in git diffs and can get very large. Instead, they are kept as only 
[jupytext](https://jupytext.readthedocs.io/en/latest/index.html) representations. 
So to build new docs, one will have to keep a local copy of every notebook executed. The
only notebooks that get committed are the ones in ``docs/recipes/``, as these require
access to large datasets and GPU resources. These are run once and committed in an 
executed format to the repository from a specific machine.

So, to contribute a notebook demonstrating or explaining some facet of *nussl*, do the
following:

1. Before you begin, navigate to the ``docs/`` directory and execute all of the notebooks present
   in the docs (this only has to happen once on your machine)::

      $ make notebooks

   This will find every ``*.py`` script in docs/examples and docs/tutorials, create the
   associated notebook, execute the cells, and convert your notebook to HTML so you
   can quickly see what it looks like without having to launch it in Jupyter notebook.
   When you run ``make html``, the executed notebooks are used in the documentation.

   To just make one notebook, run the following command::

      $ python create_and_execute_notebook path/to/script.py # make just one notebook

2. To actually create notebooks, there are two options.

   Option 1: create a new notebook in either ``tutorials`` or ``examples``. Work in the notebook
   until you are satisfied with your explanation/demo. Note: your notebook is NOT yet
   in version control!

   Now run the following command on your notebook::

      $ jupytext --set-formats ipynb,py docs/path/to/notebook.ipynb

   The light text representation of your notebook is now tracked by Git and it is 
   identical to your notebook, except for the outputs. Since your notebook is already
   executed (manually), when you make the docs you'll see it in there as long as you
   do the next step:

   Option 2: create a script that you will later link to a notebook. Work in this script
   until you're satisfied like above. Then when done with it, run::

      $ python create_and_execute_notebook path/to/script.py

   to link the script with a notebook and an HTML file. If you so wish, you can work 
   entirely in a script and just check the HTML file in your browser to make sure
   everything renders okay.

3. Add your notebook to the toctree of the associated folder. For example, 
   ``examples/primitives/primitives.rst`` has the following contents::

      Primitives
      ==========

      .. toctree::
         :maxdepth: 4

         FT2D <2dft.py>
         REPET <repet.py>
         REPETSIM <repet_sim.py>
         Melodia <melodia.py>
         TimbreClustering <timbre.py>
         HPSS <hpss.py>
   
   So if you are adding examples for a new primitive algorithm, be sure to edit 
   this like so::

      Primitives
      ==========

      .. toctree::
         :maxdepth: 4

         FT2D <2dft.py>
         REPET <repet.py>
         REPETSIM <repet_sim.py>
         Melodia <melodia.py>
         TimbreClustering <timbre.py>
         HPSS <hpss.py>
         [Your algorithm] <path/to/light/script.py>

4. Run ``make html`` from the docs folder. Then look at ``_build/html/index.html`` to see 
   what got generated. 

5. Finally, it's best to run the light script from scratch to make sure it's doing what 
   you expect::

      $ python create_and_execute_notebook path/to/your_script.py

   Inspect the resultant HTML file to make sure it's doing what you want. Then run
   ``make html`` again.


Deploying
^^^^^^^^^

This process results in rich and interactive documentation but also results in 
large files. For this reason, the actual compiled documentation is kept in a [separate
repository](https://github.com/nussl/docs), so that the main code repository stays
small. To deploy the documentation, first use the staging script. This script takes in 
an as an argument the path to the cloned documentation repository (if the path does not 
exist, then the documentation is cloned into that folder). It will first copy the contents of 
``_build/html/`` into the compiled docs repo. Then it will make a commit with update to 
the docs. Then it will tell you to push the docs after a manual inspection of it. The
sequence of commands looks like this::

      python stage_docs.py path/to/docs/repo/
      cd path/to/docs/repo
      # inspect the docs by opening index.html in your browser
      # make sure everything is okay
      git push origin master

Then visit https://nussl.github.io/docs/ to make sure everything is okay.




Adding your own algorithm
-------------------------

The one place that our process differs from other open source projects is when contributing
new algorithms to *nussl*. After your algorithm is written, the developers of *nussl* ask that you
follow these additional steps when contributing it:

0) Code passes style and error checks. I.e., *Does it follow PEP8?*, *Can this code be run on any
   machine without raising an exception?*

1) Provide benchmark files for this new algorithm. These can be one or two audio files that
   have passed through your algorithm. These files can be from established data sets or files
   that are already of the External File Zoo (EFZ). We also ask that you provide expected output values
   from one or more evaluation measures (BSS-Eval, etc).

2) If this algorithm existed somewhere else prior to this implementation, we would like a copy
   of that implementation. This will be used to benchmark against.

3) If you are NOT the original author of the algorithm, we would like written permission
   from the original author of the algorithm authorizing this implementation.

4) A reference to the academic paper that this algorithm originated from.

5) Any additional files, such as trained neural network weights, should also be provided, as these
   extra files will be needed to put on the EFZ.

If there are any questions, feel free to contact the authors via email or github.