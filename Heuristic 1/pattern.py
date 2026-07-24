"""
Stage 1 -- Pattern (DPS: Determining Partial Solution algorithm).

Generates, independently per task, an evenly-distributed accept/skip
mu-pattern that satisfies the (m,k)-firm constraint (C3, Eq. 9) -- ignoring
processor/frequency assignment entirely at this stage. This is the classic
distance-based pattern (Hamdaoui & Ramanathan): job j (1-indexed) is
accepted iff

    floor(j * m / k)  !=  floor((j - 1) * m / k)

Why this is safe: for ANY window of k consecutive jobs starting at index q,

    sum_{j=q}^{q+k-1} p(j) = floor((q+k-1)*m/k) - floor((q-1)*m/k)

telescopes to exactly m, because m is an integer and the two floor() terms
are exactly m*k/k = m apart. This holds for every possible window start q,
not just windows aligned to multiples of k -- so this pattern satisfies
the TRUE sliding-window (m,k) constraint exactly (m out of every k, no
matter where the window starts), and misses are spread as evenly as
possible by construction (no long run of consecutive misses).
"""


def generate_pattern(tasks, jobs):
    """
    tasks: list of (e, p, m, k)
    jobs:  dict i -> [1, 2, ..., eta_i]

    Returns: pattern[i][j] = 1 (accept) or 0 (skip)
    """
    pattern = {}
    for i, (e, p, m, k) in enumerate(tasks):
        pattern[i] = {}
        for j in jobs[i]:
            count_j = (j * m) // k
            count_j_minus_1 = ((j - 1) * m) // k
            pattern[i][j] = 1 if count_j != count_j_minus_1 else 0
    return pattern
