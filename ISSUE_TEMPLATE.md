<!--
NOTE: Stolen from 
-->

<!-- Instructions For Filing a Bug: https://github.com/interactiveaudiolab/nussl/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
<!-- Example: Calling repet.run() twice after specifying min,max period causes error -->

#### Steps/Code to Reproduce
<!--
Running repet twice with the same AudioSignal object with provided period range crashes.
Example:
```python
repet = nussl.Repet(signal1, min_period=3.4, max_period=3.6)
repet()
repet()
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Versions
<!--
Please run the following snippet and paste the output below.
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import librosa; print("librosa", librosa.__version__)
import nussl; print("nussl", nussl.__version__)
-->


<!-- Thanks for contributing! -->
