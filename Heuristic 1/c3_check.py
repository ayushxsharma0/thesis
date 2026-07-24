"""
C3 verification and repair.

Stage 1's distance-based pattern satisfies (m,k) EXACTLY (m out of every k,
no slack) for every sliding window. This means Stage 2 (SPS) demoting even
a single accepted job breaks C3 for every window containing it, since
there is zero margin to absorb a loss.

find_violations() detects this. repair_violations() attempts to fix it by
trying to place ANY currently-skipped job that falls inside a violated
window for that task -- not necessarily the exact job Stage 1 originally
picked, since Stage 1's pattern only tells us HOW MANY jobs per window
are required (m), not that it must be those specific m jobs.

This repair pass must run BEFORE the general fairness-improvement loop in
fairness.py, since restoring correctness (C3) takes priority over
optimizing the objective (min-max gamma) on top of a correct solution.
"""


def find_violations(state, tasks, jobs, eta):
    """
    Returns a list of (task_index, window_start_job, window_jobs) for every
    sliding window of size k whose accepted count is currently < m.
    """
    violations = []

    for i, (e, p, m, k) in enumerate(tasks):
        n = eta[i]
        if n < k:
            continue

        for start in range(1, n - k + 2):
            window_jobs = list(range(start, start + k))
            accepted_count = sum(
                1 for j in window_jobs if state.assignment[(i, j)] is not None
            )
            if accepted_count < m:
                violations.append((i, start, window_jobs))

    return violations


def repair_violations(state, tasks, jobs, eta, freq_order=None):
    """
    Attempts to fix every C3 violation by placing additional jobs from
    within the violated windows. Repeats until either no violations remain
    or a full pass fixes nothing (meaning remaining violations are
    infeasible given current DBF/energy commitments and processing order).

    Returns: (state, unresolved_violations)
    """
    while True:
        violations = find_violations(state, tasks, jobs, eta)
        if not violations:
            return state, []

        fixed_any = False

        for (i, start, window_jobs) in violations:
            m = tasks[i][2]
            accepted_count = sum(
                1 for j in window_jobs if state.assignment[(i, j)] is not None
            )
            still_needed = m - accepted_count
            if still_needed <= 0:
                continue  # may have been fixed by an earlier repair in this pass

            skipped_in_window = [
                j for j in window_jobs if state.assignment[(i, j)] is None
            ]
            # earliest deadline first among candidates in this window
            skipped_in_window.sort(key=lambda j: j * tasks[i][1])

            for j in skipped_in_window:
                if still_needed <= 0:
                    break
                if state.try_place_best(i, j, freq_order=freq_order):
                    still_needed -= 1
                    fixed_any = True

        if not fixed_any:
            # no progress this pass -- remaining violations are stuck
            # given current DBF/energy commitments and the order jobs
            # were placed in.
            return state, find_violations(state, tasks, jobs, eta)
