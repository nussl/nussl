import nussl
import pytest
import numpy as np
from nussl.separation.base import SeparationException
from nussl.core.migration import SafeModelLoader
from copy import deepcopy


fix_dir = 'tests/local/trainer'

def test_safe_model_loader():
    safe_loader = SafeModelLoader()

    with pytest.raises(SeparationException):
        safe_loader._get_moved({}, {}, '')

    assert 'nussl_version' in safe_loader._get_moved({'nussl_version': 1}, {}, '')
    assert 'nussl_version' in safe_loader._get_moved({}, {'nussl_version': 1}, '')
    assert 'config' in safe_loader._get_moved({'nussl_version': 1}, {'config': {}}, '')

    with pytest.raises(SeparationException):
        safe_loader._load_types(str, '', 123)

    assert safe_loader._load_types(str, '', None) == 'UNAVAILABLE'
    assert safe_loader._load_types(str, '', 'hello') == 'hello'
    assert safe_loader._load_types(int, '', 123) == 123

    safe_loader.eval = 'BssEvalScale'
    safe_loader._load_eval({})
    safe_loader.eval = 'BSSEvalV4'
    safe_loader._load_eval({})
