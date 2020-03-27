"""
Evaluation
==========

Evaluation base
---------------

.. autoclass:: nussl.evaluation.EvaluationBase
    :members:
    :autosummary:

BSS Evaluation base
-------------------

.. autoclass:: nussl.evaluation.BSSEvaluationBase
    :members:
    :autosummary:

Scale invariant BSSEval
-----------------------

.. autoclass:: nussl.evaluation.BSSEvalScale
    :members:
    :autosummary:

.. autofunction:: nussl.evaluation.scale_bss_eval

BSSEvalV4 (museval)
-------------------

.. autoclass:: nussl.evaluation.BSSEvalV4
    :members:
    :autosummary:

Precision and recall on masks
-----------------------------

.. autoclass:: nussl.evaluation.PrecisionRecallFScore
    :members:
    :autosummary:

"""

from .evaluation_base import EvaluationBase, aggregate_score_files
from .bss_eval import BSSEvaluationBase, BSSEvalV4, BSSEvalScale, scale_bss_eval
from .precision_recall_fscore import PrecisionRecallFScore
