from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='nussl',
    version='1.0.0',
    classifiers=[
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
    ],
    description='A flexible sound source separation library.',
    author='E. Manilow, P. Seetharaman, F. Pishdadian, N. Shelly, B. Pardo',
    author_email='ethanmanilow@u.northwestern.edu',
    maintainer='E. Manilow, P. Seetharaman, F. Pishdadian, N. Shelly, B. Pardo',
    maintainer_email='ethanmanilow@u.northwestern.edu',
    url='https://github.com/interactiveaudiolab/nussl',
    license='MIT',
    packages=['nussl'],
    keywords=['audio', 'source', 'separation', 'music', 'sound', 'source separation'],
    install_requires=[
        'jams',
        'librosa',
        'matplotlib',
        'mir_eval',
        'museval',
        'musdb',
        'pyyaml',
        'zarr==2.3.0',
        'numcodecs==0.6.2',
        'ffmpy',
        'torch',
        'pytorch-ignite',
        'tensorboard,
        'norbert'
    ],
    extras_require={
        'melodia': [
            'vamp'
        ],
        'tests': ['pytest', 'pytest_cov']
    }
)