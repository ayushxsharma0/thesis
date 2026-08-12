"""
periods.py -- harmonic period generation.

Generates task periods in STRICT geometric progression:
    p_i = p_min * ratio^(i-1)

This satisfies the paper's harmonic condition p_{i+1}/p_i = p_{i+2}/p_{i+1}
= ratio for every i (constant ratio between consecutive periods, not just
pairwise divisibility).

NOTE on scale: the effective hyperperiod h_eff grows roughly as
2 * p_max = 2 * p_min * ratio^(n_tsk-1), which is EXPONENTIAL in n_tsk.
Keep ratio small (2 or 3) and n_tsk modest unless a very large instance
is deliberately wanted -- this module does not itself cap anything; the
master file (generate_testcase.py) is responsible for rejecting/retrying
draws that produce an unreasonably large h_eff.
"""


def generate_harmonic_periods(n_tsk, p_min, ratio):
    """
    n_tsk: number of tasks
    p_min: base (smallest) period, an integer
    ratio: integer growth factor between consecutive periods

    Returns: list of n_tsk periods, e.g. [p_min, p_min*ratio, p_min*ratio^2, ...]
    """
    return [p_min * (ratio ** i) for i in range(n_tsk)]