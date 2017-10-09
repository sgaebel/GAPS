#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='gaps',
      version='0.1',
      description='GPU Accelerated Parallel Sampler',
      author='Sebastian M. Gaebel',
      author_email='sebastian.gaebel@ligo.org',
      license='MIT',
      url='https://github.com/sgaebel/GAPS',
      packages=['gaps'],
      install_requires=['pyopencl', 'numpy', 'matplotlib'],
      extras_require={'test': 'scipy'})
