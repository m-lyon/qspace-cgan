#!/usr/bin/env python3
'''Use this to install module'''
from os import path
from setuptools import setup, find_namespace_packages

install_deps = []

version = '1.0.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qspace-cgan',
    version=version,
    description='q-space conditioned CGAN.',
    author='Matt Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=install_deps,
    scripts=[],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows :: Windows 10',
    ],
)
