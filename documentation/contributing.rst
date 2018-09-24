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