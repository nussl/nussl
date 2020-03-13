from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('extra_requirements.txt') as f:
    extra_requirements = f.read().splitlines()

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
        'Programming Language :: Python :: 3.7',
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
    packages=find_packages(),
    keywords=['audio', 'source', 'separation', 'music', 'sound', 'source separation'],
    install_requires=requirements,
    extras_require={
        'melodia': [
            'vamp'
        ],
        'extras': extra_requirements
    }
)