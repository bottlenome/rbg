#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os

from setuptools import setup, find_packages

try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''


def _requires_from_file(filename):
    return open(filename).read().splitlines()


# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace("'", '')
                for line in open(os.path.join(here,
                                              'rbg',
                                              '__init__.py'))
                if line.startswith('__version__ = ')),
               '0.0.0')

setup(
    name="rbg",
    version=version,
    url='https://github.com/bottlenome/rbg',
    author='bottlenome',
    author_email='bottlenome@gmail.com',
    maintainer='bottlenome',
    maintainer_email='bottlenome@gmail.com',
    description='Random Batch Generalization: in model augmentation layer',
    long_description=readme,
    keywords='deeplerning, pytorch',
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    license="Creative Commons Zero v1.0 Universal",
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    ],
)
