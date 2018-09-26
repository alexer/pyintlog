#! /usr/bin/env python
try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

setup(
	name = 'intlog',
	version = '1.0',
	description = 'Functions for calculating integer logarithms',
	author = 'Aleksi Torhamo',
	author_email = 'aleksi@torhamo.net',
	url = 'http://github.com/alexer/intlog',
	py_modules = ['intlog'],
)

