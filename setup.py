#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install import INSTALL_SCHEMES
from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'swcsim'
VERSION =            '0.1'
AUTHOR =             'Chung-Heng Yeh'
AUTHOR_EMAIL =       'chungheng.yeh@gmail.com'
URL =                'https://github.com/chungheng/SWCSimulator'
MAINTAINER =         AUTHOR
MAINTAINER_EMAIL =   AUTHOR_EMAIL
DESCRIPTION =        'SWC Simulator'
LONG_DESCRIPTION =   'A python library for simulating neuron models specified in the SWC format'
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

# Explicitly switch to parent directory of setup.py in case it
# is run from elsewhere:
os.chdir(os.path.dirname(os.path.realpath(__file__)))
PACKAGES = find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages=PACKAGES,
        install_requires=[
            'recordclass >= 0.4.3',
            'networkx >= 1.11',
            'pandas >= 0.17.1'
        ]
    )
