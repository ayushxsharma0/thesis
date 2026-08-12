"""
processors.py -- processor-side generation (frequency levels).

Generates n_frq evenly spaced frequency levels between f_min and f_max
(inclusive).

NOTE on scope: n_prc itself needs no generation logic -- it's a single
integer sampled directly from config (e.g. a random pick from
[1,2,4,8,16]), which is a config-notation concern handled by the master
file. This module covers the one processor-side quantity that actually
requires computation: turning (n_frq, f_min, f_max) into a concrete list
of frequency levels.
"""


def generate_frequency_levels(n_frq, f_min, f_max):
    """
    Returns: list of n_frq evenly spaced levels in [f_min, f_max].
    """
    if n_frq == 1:
        return [f_max]
    step = (f_max - f_min) / (n_frq - 1)
    return [round(f_min + i * step, 3) for i in range(n_frq)]