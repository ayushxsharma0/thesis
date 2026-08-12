"""
mk_pattern.py -- (m,k) constraint generation.

Samples (m, k) for a single task. k is drawn uniformly from [k_min, k_max].
m is derived from an m/k ratio drawn from a sub-band of [ratio_lo, ratio_hi]
depending on mk_type:
    dense  -> upper third of the range (few misses tolerated)
    sparse -> lower third of the range (many misses tolerated)
    mixed  -> full range

NOTE on scope: which mk_type to use for a given testcase (random pick
from [dense, sparse, mixed]) and the overall [ratio_lo, ratio_hi] bounds
(parsed from config as fixed sub-band boundaries) are config-notation
concerns, handled by the master file. This module only generates one
task's (m,k) given an already-chosen mk_type and already-resolved bounds.
"""

import random


def sample_mk_for_task(mk_type, ratio_lo, ratio_hi, k_min, k_max):
    """
    Uses the global `random` module (seeded by the master file before
    calling this), consistent with the other worker modules.

    Returns: (m, k)
    """
    k = random.randint(k_min, k_max)
    third = (ratio_hi - ratio_lo) / 3.0

    if mk_type == "dense":
        lo, hi = ratio_hi - third, ratio_hi
    elif mk_type == "sparse":
        lo, hi = ratio_lo, ratio_lo + third
    else:  # mixed
        lo, hi = ratio_lo, ratio_hi

    ratio = random.uniform(lo, hi)
    m = max(1, min(k, round(ratio * k)))
    return m, k