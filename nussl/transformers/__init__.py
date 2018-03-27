#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
                          /[-])//  ___
                     __ --\ `_/~--|  / \
                   /_-/~~--~~ /~~~\\_\ /\
                   |  |___|===|_-- | \ \ \
 _/~~~~~~~~|~~\,   ---|---\___/----|  \/\-\
 ~\________|__/   / // \__ |  ||  / | |   | |
          ,~-|~~~~~\--, | \|--|/~|||  |   | |
          [3-|____---~~ _--'==;/ _,   |   |_|
                      /   /\__|_/  \  \__/--/
                     /---/_\  -___/ |  /,--|
                     /  /\/~--|   | |  \///
                    /  / |-__ \    |/
                   |--/ /      |-- | \
                  \^~~\\/\      \   \/- _
                   \    |  \     |~~\~~| \
                    \    \  \     \   \  | \
                      \    \ |     \   \    \
                       |~~|\/\|     \   \   |
                      |   |/         \_--_- |\
                      |  /            /   |/\/
                       ~~             /  /
                                     |__/

Imports for transformers classes. More than meets the eye.
"""

from .transformer_nmf import TransformerNMF
from .transformer_deep_clustering import TransformerDeepClustering, show_model, affinity_cost

__all__ = ['TransformerNMF', 'TransformerDeepClustering', 'show_model', 'affinity_cost']
