#!/usr/bin/env python

import os
import sys

import subprocess
subprocess.run('poetry install', shell=True)

#old attempt
# import setuptools
# import setuptools.command
#
# sys.path.append(os.path.dirname(__file__))
#
#
# # TODO: UPDATE METADATA
# setuptools.setup(
#     name="ssrm_test",
#     version="0.1.4",
#     description="CAUTION: NONSTANDARD INSTALL! A library for Sequential Sample Ratio Mismatch (SRM) test.",
#     license="Apache-2.0",
#     readme="README.md",
#     author="Michael Lindon",
#     author_email="michael.lindon@optimizely.com",
#     url="https://github.com/optimizely/ssrm",
#     package_dir={"": "ssrm_test"},
#     package_data={"": ["*.html", "*.css"]},
#     include_package_data=True,
#     packages=setuptools.find_packages(where="ssrm_test"),
#     classifiers=[
#         "Development Status :: 3 - Alpha",
#         "License :: Other/Proprietary License",
#         "Programming Language :: Python :: 3 :: Only",
#         "Topic :: Software Development :: Libraries :: Python Modules",
#         "Typing :: Typed",
#     ],
#     install_requires=[
#         "numpy~=1.16",
#         "scipy~=1.3",
#         "toolz~=0.10.0",
#         "typing~=3.7.4",
#         "poetry~=0.12",
#     ],
# )
