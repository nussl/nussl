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

Aggregators
-----------

.. autofunction:: nussl.evaluation.aggregate_score_files


"""

from .report_card import aggregate_score_files, report_card, associate_metrics
from .evaluation_base import EvaluationBase
from .bss_eval import BSSEvaluationBase, BSSEvalV4, BSSEvalScale, scale_bss_eval
from .precision_recall_fscore import PrecisionRecallFScore
