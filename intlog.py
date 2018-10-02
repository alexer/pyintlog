# -!- encoding: utf-8 -!-
"""
Functions for calculating integer logarithms

This module provides functions for calculating 4 different variants of integer logarithm, which differ on how the
real-valued logarithm is rounded to an integer and correspond to the inequality operators >, ≥, <, and ≤ (based on
how the integer return value corresponds to the real-valued return value):

exponent = gt_log(value, base) = ceil(log(value + 1, base))   # base ** (exponent - 1) <= value <  base ** exponent
exponent = ge_log(value, base) = ceil(log(value, base))       # base ** (exponent - 1) <  value <= base ** exponent
exponent = lt_log(value, base) = floor(log(value - 1, base))  # base ** exponent       <  value <= base ** (exponent + 1)
exponent = le_log(value, base) = floor(log(value, base))      # base ** exponent       <= value <  base ** (exponent + 1)

There's also an int_log(value, base, rounding), which selects the variant based on the rounding argument.

Of course, because of the inexactness of floating point, the definitions above - which just ceil/floor the return
value of the normal log() function - can't be used directly for calculating the integer logarithm. For example, on
many environments, log(125, 5) returns 3.0000000000000004 instead of 3, which means that gt_log(124, 5) - if
implemented as ceil(log(value + 1, base)) - would return 4 instead of the correct 3.

Most functions in this module use a lookup table to speed up the calculation, so their time complexity is O(1)
instead of the usual O(log N). The fast_*_log() functions are strictly O(1), but require an explicit call to
extend_fast_intlog_range_to_*() - which is O(log N) - before the first calculation, to initialize the
aforementioned lookup table. The *_log() functions extend the range automatically as required, so their complexity
is amortized O(1). The slow_*_log() functions use the usual algorithm, so their complexity is O(log N).
(The point of the slow_*_log() functions is that if you know that your value is going to be absolutely humongous,
and you care more about memory than speed, you can avoid filling the lookup table if you want)

The space requirement of the lookup table is O(log N) per each used base; Logarithms of values up to 2**64 in any
base can be calculated with a lookup table of around 64 elements. The LUT method is 2-10x faster than the usual
method on typical inputs, although we are talking about only microseconds here. (The usual method can be ~30% - or
a fraction of a microsecond - faster when the given value is smaller than the base itself)

Note: You must call extend_fast_intlog_range_to_*() before the first call to any of the fast_*_log() functions.
"""
from __future__ import unicode_literals, print_function
import operator

# Mapping from the rounding argument of int_log() to the needed operations
_ROUNDING = {
	'gt': (operator.add, operator.ge, operator.le, 0),
	'ge': (operator.add, operator.gt, operator.lt, 0),
	'lt': (operator.sub, operator.le, operator.lt, 1),
	'le': (operator.sub, operator.lt, operator.le, 1),
}

# Allow the comparison functions from the operator module to be used in place of strings
for name, value in list(_ROUNDING.items()):
	_ROUNDING[getattr(operator, name)] = value

# Allow the symbols for the comparison functions to be used in place of the abbreviations
for item in '>:gt >=:ge ≥:ge <:lt <=:le ≤:le'.split():
	alias, name = item.split(':')
	_ROUNDING[alias] = _ROUNDING[name]

del operator, name, value, item, alias


_log_error = 'Logarithm is only defined for numbers greater than zero (the logarithm approaches negative infinity as the argument approaches zero)'


_lut = {}
def _init_base(base):
	"""Validate the given base, and either initialize or fetch the lookup table - and the previous exponent and power - for it"""
	if not isinstance(base, int):
		raise TypeError('Integer logarithm base must be an integer, not %s' % base.__class__.__name__)
	if base <= 1:
		raise ValueError('Integer logarithm base must be greater than one')
	limits = _lut.setdefault(base, [None, (0, 1)])
	return (limits,) + limits[-1]

def _init_next_power(limits, exponent, power):
	"""Insert LUT entries for all bit lengths between the given power and the previous one

	The given exponent must be exactly one larger than the previous one!
	"""
	bitlen = power.bit_length()
	# All numbers (with previous_bitlen < number.bit_length() <= bitlen) less than the given power have ceil(log(number)) equal to the given exponent
	while len(limits) <= bitlen:
		limits.append((exponent, power))
	return bitlen

def extend_fast_intlog_range_to_power(max_exponent, base):
	"""Initialize or extend the range supported by the fast functions so that all integers up to base ** max_exponent are supported for the given base"""
	limits, exponent, power = _init_base(base)
	for exponent in range(exponent + 1, max_exponent + 1):
		power *= base
		_init_next_power(limits, exponent, power)
	return exponent, power

def extend_fast_intlog_range_to_bitlen(max_bitlen, base):
	"""Initialize or extend the range supported by the fast functions so that all integers up to the given bit length are supported for the given base"""
	limits, exponent, power = _init_base(base)
	bitlen = len(limits) - 1
	while bitlen < max_bitlen:
		power *= base
		exponent += 1
		bitlen = _init_next_power(limits, exponent, power)
	return exponent, power


def gt_log(value, base):
	"""Return the minimum exponent of base such that the power is greater than value

	That is, the returned exponent satisfies: base ** (exponent - 1) <= value < base ** exponent
	This is also the exact integer equivalent of: ceil(log(value + 1, base))
	"""
	if value <= 0:
		raise ValueError(_log_error)
	try:
		exponent, power = _lut[base][value.bit_length()]
	except (KeyError, IndexError):
		exponent, power = extend_fast_intlog_range_to_bitlen(value.bit_length(), base)
	return exponent + (value >= power)

def ge_log(value, base):
	"""Return the minimum exponent of base such that the power is greater than or equal to value

	That is, the returned exponent satisfies: base ** (exponent - 1) < value <= base ** exponent
	This is also the exact integer equivalent of: ceil(log(value, base))
	"""
	if value <= 0:
		raise ValueError(_log_error)
	try:
		exponent, power = _lut[base][value.bit_length()]
	except (KeyError, IndexError):
		exponent, power = extend_fast_intlog_range_to_bitlen(value.bit_length(), base)
	return exponent + (value > power)

def lt_log(value, base):
	"""Return the maximum exponent of base such that the power is less than value

	That is, the returned exponent satisfies: base ** exponent < value <= base ** (exponent + 1)
	This is also the exact integer equivalent of: floor(log(value - 1, base))
	"""
	if value <= 0:
		raise ValueError(_log_error)
	try:
		exponent, power = _lut[base][value.bit_length()]
	except (KeyError, IndexError):
		exponent, power = extend_fast_intlog_range_to_bitlen(value.bit_length(), base)
	return exponent - (value <= power)

def le_log(value, base):
	"""Return the maximum exponent of base such that the power is less than or equal to value

	That is, the returned exponent satisfies: base ** exponent <= value < base ** (exponent + 1)
	This is also the exact integer equivalent of: floor(log(value, base))
	"""
	if value <= 0:
		raise ValueError(_log_error)
	try:
		exponent, power = _lut[base][value.bit_length()]
	except (KeyError, IndexError):
		exponent, power = extend_fast_intlog_range_to_bitlen(value.bit_length(), base)
	return exponent - (value < power)

def int_log(value, base, rounding):
	"""Return the integer logarithm of value for the given base according to the given rounding

	The rounding argument should be one of the comparison operators >, ≥, <, or ≤; It determines how
	the returned integer is chosen in relation to the real-valued return value of log(value, base).
	"""
	if value <= 0:
		raise ValueError(_log_error)
	add_op, cmp_op, _, _ = _ROUNDING[rounding]
	try:
		exponent, power = _lut[base][value.bit_length()]
	except (KeyError, IndexError):
		exponent, power = extend_fast_intlog_range_to_bitlen(value.bit_length(), base)
	return add_op(exponent, cmp_op(value, power))


def fast_gt_log(value, base):
	exponent, power = _lut[base][value.bit_length()]
	return exponent + (value >= power)

def fast_ge_log(value, base):
	exponent, power = _lut[base][value.bit_length()]
	return exponent + (value > power)

def fast_lt_log(value, base):
	exponent, power = _lut[base][value.bit_length()]
	return exponent - (value <= power)

def fast_le_log(value, base):
	exponent, power = _lut[base][value.bit_length()]
	return exponent - (value < power)

def fast_int_log(value, base, rounding):
	add_op, cmp_op, _, _ = _ROUNDING[rounding]
	exponent, power = _lut[base][value.bit_length()]
	return add_op(exponent, cmp_op(value, power))


def slow_gt_log(value, base):
	if value <= 0:
		raise ValueError(_log_error)
	power = 1
	exponent = 0
	while power <= value:
		power *= base
		exponent += 1
	return exponent

def slow_ge_log(value, base):
	if value <= 0:
		raise ValueError(_log_error)
	power = 1
	exponent = 0
	while power < value:
		power *= base
		exponent += 1
	return exponent

def slow_lt_log(value, base):
	if value <= 0:
		raise ValueError(_log_error)
	power = 1
	exponent = 0
	while power < value:
		power *= base
		exponent += 1
	return exponent - 1

def slow_le_log(value, base):
	if value <= 0:
		raise ValueError(_log_error)
	power = 1
	exponent = 0
	while power <= value:
		power *= base
		exponent += 1
	return exponent - 1

def slow_int_log(value, base, rounding):
	if value <= 0:
		raise ValueError(_log_error)
	_, _, cmp_op, sub = _ROUNDING[rounding]
	power = 1
	exponent = 0
	while cmp_op(power, value):
		power *= base
		exponent += 1
	return exponent - sub


def _copy_docstrings(**adfixes):
	"""Copy docstrings from the default functions to the given variants, with the given extra documentation"""
	globs = globals()
	for func_prefix, doc_suffix in adfixes.items():
		for variant in 'gt ge lt le int'.split():
			src_name = '_'.join((variant, 'log'))
			dst_name = '_'.join((func_prefix, variant, 'log'))
			doc = globs[src_name].__doc__
			if doc_suffix:
				delim = '\n' + doc.rsplit('\n')[-1]
				doc = delim.join((doc, doc_suffix, ''))
			globs[dst_name].__doc__ = doc

_copy_docstrings(
	fast='Note: You must call extend_fast_intlog_range_to_*() before the first call to this function.',
	slow='',
)


def _test_intlog_funcs(funcs, max_value, bases, precision=14, extend_range=False, verbose=False):
	"""Test the given integer logarithm functions up to max_value for all the given bases

	A note on the precision argument:
	Since some of the floating point values returned by log(base ** exponent, base) are not exactly equal to exponent, we round
	the return value of log() to the given precision - just enough to get the correct return value with exact powers.
	"""
	from math import log, ceil, floor

	# When we say ceil(log(value + 1, base)), we really mean ceil(log(value + epsilon, base)), where 0 < epsilon <= 1
	# Since log(0) is undefined, we use epsilon < 1, so we don't get an error in the floor(log(value - epsilon, base)) case
	epsilon = 0.5

	gt_func, ge_func, lt_func, le_func = funcs
	for base in bases:
		if verbose:
			print('Testing base', base)
		if extend_range:
			extend_fast_intlog_range_to_bitlen(max_value.bit_length(), base)
		for value in range(1, max_value + 1):
			exponent = gt_func(value, base)
			assert base ** (exponent - 1) <= value < base ** exponent
			assert exponent == ceil(round(log(value + epsilon, base), precision))
			exponent = ge_func(value, base)
			assert base ** (exponent - 1) < value <= base ** exponent
			assert exponent == ceil(round(log(value, base), precision))
			exponent = lt_func(value, base)
			assert base ** exponent < value <= base ** (exponent + 1)
			assert exponent == floor(round(log(value - epsilon, base), precision))
			exponent = le_func(value, base)
			assert base ** exponent <= value < base ** (exponent + 1)
			assert exponent == floor(round(log(value, base), precision))

def _test_all_intlog_funcs(max_value, bases, precision=14, verbose=False):
	from functools import partial

	globs = globals()
	ops = 'gt ge lt le'.split()
	test_funcs = partial(_test_intlog_funcs, max_value=max_value, bases=bases, precision=precision, verbose=verbose)

	for speed in 'slow_  fast_'.split(' '):
		funcs = [globs[speed + op + '_log'] for op in ops]
		if verbose:
			print('Testing functions:', ' '.join(func.__name__ for func in funcs))
		test_funcs(funcs)

		func = globs[speed + 'int_log']
		if verbose:
			print('Testing function:', func.__name__)
		test_funcs([partial(func, rounding=op) for op in ops])

if __name__ == '__main__':
	import sys

	max_value = 10**3
	max_base = 10
	try:
		assert not len(sys.argv) > 3
		if len(sys.argv) > 1:
			max_value = eval(sys.argv[1])
		if len(sys.argv) > 2:
			max_base = int(sys.argv[2])
	except:
		print('Error: Invalid argument', file=sys.stderr)
		print('Usage: python -m intlog [max_value [max_base]]', file=sys.stderr)
		sys.exit(-1)

	_test_all_intlog_funcs(max_value, range(2, max_base + 1), verbose=True)

