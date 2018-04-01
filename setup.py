#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from imp import load_source
import os
import codecs
import re

try:
    from setuptools import setup
except:
    from distutils.core import setup

######################################################

NAME = 'nussl'
META_PATH = os.path.join('nussl', '__init__.py')
KEYWORDS = ['audio', 'source', 'separation', 'music', 'sound', 'source separation']
REQUIREMENTS = [
    'numpy >= 1.7.0',
    'scipy >= 0.12.0',
    'matplotlib >= 1.3.0',
    'audioread >= 2.1.2',
    'librosa >= 0.4.1',
    'mir_eval >= 0.4.0',
    'sklearn'
]

EXTRAS = {
    'melodia': [
        'vamp'
    ],
    'deep': [
        'torch'
    ],
    'musdb': [
        'stempeg'
    ]
}

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.7',
    'Topic :: Artistic Software',
    'Topic :: Multimedia',
    'Topic :: Multimedia :: Sound/Audio',
    'Topic :: Multimedia :: Sound/Audio :: Editors',
    'Topic :: Software Development :: Libraries',
]

######################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == '__main__':
    setup(
        name='nussl',
        version=find_meta('version'),
        classifiers=CLASSIFIERS,
        description=find_meta('description'),
        author=find_meta('author'),
        author_email=find_meta('email'),
        maintainer=find_meta('author'),
        maintainer_email=find_meta('email'),
        url=find_meta('uri'),
        license=find_meta('license'),
        packages=find_packages(),
        keywords=KEYWORDS,
        install_requires=REQUIREMENTS,
        extras_require=EXTRAS
    )
