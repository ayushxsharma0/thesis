"""
Stage 3 -- Fairness (greedy improvement).

Starting from the feasible mu_p-pattern produced by SPS, repeatedly try to
flip currently-skipped jobs to accepted, always targeting the task with
the current WORST miss ratio (max gamma_i, Eq. 3/4) first.

NOTE on tie-breaking: eta_i is a TASK-level parameter -- every job of a
given task shares the same e_i, so "smallest execution time" cannot
distinguish between two skipped jobs of the SAME task (they tie). Once the
worst task is selected, we break ties by earliest absolute deadline (EDF
order), which is the natural default. If the task model is later extended
to allow per-job execution time variance, swap the sort key in
_ordered_skipped_jobs() to sort by e_{i,j} directly.

C3 ((m,k)) safety: every flip only ADDS an accepted job on top of the
Stage-1 baseline pattern, never removes one, so each task's accepted-job
count can only stay the same or increase relative to a pattern that
already satisfied C3 for every sliding window -- C3 remains satisfied
throughout.

Each flip attempt is checked against C2 (DBF) and C4 (energy) via
SchedulerState.try_place_best; if infeasible, that job is left skipped and
the next candidate is tried (per spec: skip, don't retry with a different
frequency/processor combo beyond what try_place_best already searches).

Terminates when a full pass over all tasks produces no successful flip.
"""


def _gamma(state, jobs, i, eta_i):
    accepted = sum(1 for j in jobs[i] if state.assignment[(i, j)] is not None)
    return (eta_i - accepted) / eta_i


def _ordered_skipped_jobs(state, tasks, jobs, i):
    """Skipped jobs of task i, earliest deadline first (EDF tie-break)."""
    p = tasks[i][1]
    skipped = [j for j in jobs[i] if state.assignment[(i, j)] is None]
    skipped.sort(key=lambda j: j * p)
    return skipped


def run_fairness(state, tasks, jobs, eta):
    n_tsk = len(tasks)

    # --- Mandatory C3 repair pass -----------------------------------
    # SPS (Stage 2) may have demoted Stage-1-accepted jobs to fit DBF/
    # energy constraints. Because Stage 1's pattern has EXACTLY m
    # accepted per window (zero slack), any such demotion breaks C3 for
    # every window containing that job. Restoring C3 is a correctness
    # requirement and must happen before we optimize fairness on top of
    # a possibly-invalid schedule.
    from c3_check import repair_violations

    high_freq_first = list(range(len(state.freq_levels) - 1, -1, -1))
    state, unresolved = repair_violations(
        state, tasks, jobs, eta, freq_order=high_freq_first
    )

    if unresolved:
        print(f"[Fairness] WARNING: {len(unresolved)} C3 (m,k) violation(s) "
              f"could not be repaired given current DBF/energy commitments. "
              f"This instance may be infeasible, or a different SPS "
              f"processing order might avoid the conflict.")
        for (i, start, window_jobs) in unresolved:
            print(f"    Task {i}, window starting job {start}: "
                  f"jobs {window_jobs}")
    # ------------------------------------------------------------------

    improved = True

    while improved:
        improved = False

        # rank tasks by current miss ratio, worst first
        gammas = [(_gamma(state, jobs, i, eta[i]), i) for i in range(n_tsk)]
        gammas.sort(reverse=True)

        for gamma_i, i in gammas:
            if gamma_i <= 0:
                continue  # this task already has zero misses

            # Highest frequency first: minimizes DBF demand per job, which
            # maximizes the chance a flip succeeds. Stage 3's goal is
            # squeezing in extra accepted jobs for fairness, not saving
            # energy -- that trade-off already happened in Stage 2.
            freq_order_high_first = list(range(len(state.freq_levels) - 1, -1, -1))

            for j in _ordered_skipped_jobs(state, tasks, jobs, i):
                if state.try_place_best(i, j, freq_order=freq_order_high_first):
                    improved = True
                    break  # re-rank tasks from scratch after every flip

            if improved:
                break

    return state