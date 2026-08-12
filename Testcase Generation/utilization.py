"""
utilization.py -- UUniFast-Discard utilization splitting.

Splits a total utilization value U across n tasks, grounded in Bini &
Buttazzo's UUniFast algorithm, with the "discard if any u_i > 1"
extension needed for multiprocessor settings, since no single task may
require more than one full processor's worth of continuous capacity.

NOTE on scope: which total utilization value U to use (e.g. picking a
band like [0.2-0.4] and sampling within it) is a config-notation
concern, handled by the master file (generate_testcase.py). This module
only does the split of an already-chosen U across n tasks.
"""

import random


def uunifast_discard(n, U, u_max=1.0, max_attempts=1000):
    """
    Generate n utilization values summing to U, each in (0, u_max].
    Retries (discards) whenever any generated value exceeds u_max.

    Uses the global `random` module (seeded by the master file before
    calling this), consistent with the other worker modules.
    """
    for _ in range(max_attempts):
        utils = []
        sum_u = U
        ok = True
        for i in range(1, n):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
            u_i = sum_u - next_sum_u
            if u_i > u_max:
                ok = False
                break
            utils.append(u_i)
            sum_u = next_sum_u
        if ok and sum_u <= u_max:
            utils.append(sum_u)
            return utils
    raise RuntimeError(
        f"UUniFast-Discard failed to generate a valid utilization vector "
        f"after {max_attempts} attempts (U={U}, n={n}, u_max={u_max}). "
        f"Try a lower total utilization U or more tasks n."
    )