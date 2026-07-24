"""
Stage 2 -- SPS (Sum of Partial Solutions).

Combines the per-task partial patterns from Stage 1 into a single,
processor/frequency-feasible mu_p-pattern. Jobs Stage 1 marked "accepted"
are scheduled in EDF (earliest absolute deadline first) order onto
processors, at the lowest frequency that keeps them DBF-feasible (C2,
Eq. 7-8) and within the energy budget (C4, Eq. 10).

If no processor/frequency combination works for a job, it is demoted back
to "skipped" -- Stage 1 only guarantees per-task (m,k) feasibility in
isolation; it says nothing about cross-task DBF/energy feasibility once
all tasks share the same processors. This "summing" of independently
generated partial solutions into one jointly-feasible solution is what
gives the stage its name.
"""

from feasibility import SchedulerState


def run_sps(tasks, jobs, pattern, time_pairs, energy_budget, freq_levels,
            n_prc, h_bar=1.0):
    state = SchedulerState(tasks, jobs, time_pairs, energy_budget,
                            freq_levels, n_prc, h_bar)

    # collect jobs Stage 1 marked as accepted
    accepted_jobs = [
        (i, j)
        for i in range(len(tasks))
        for j in jobs[i]
        if pattern[i][j] == 1
    ]

    # EDF order: sort by absolute deadline j * p_i
    accepted_jobs.sort(key=lambda ij: ij[1] * tasks[ij[0]][1])

    demoted = []
    for (i, j) in accepted_jobs:
        if not state.try_place_best(i, j):
            demoted.append((i, j))

    return state, demoted
