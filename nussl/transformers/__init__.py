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

from .. import torch_imported, ImportErrorClass
if torch_imported:
    from .transformer_deep_clustering import TransformerDeepClustering
else:
    class TransformerDeepClustering(ImportErrorClass):
        def __init__(self):
            super(TransformerDeepClustering, self).__init__('pytorch')


__all__ = ['TransformerNMF', 'TransformerDeepClustering']
