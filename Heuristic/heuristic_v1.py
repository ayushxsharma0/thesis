"""
heuristic1.py -- Heuristic 1 (initial version, devised with guide).

Pipeline:
  1. E-pattern generation: evenly-distributed distance-based mu-pattern
     per task, satisfying (m,k) exactly for every sliding window.
  2. SPS: combine all per-task patterns into one processor/frequency-
     assigned schedule (EDF order, demoting jobs that don't fit once
     tasks share processors).
  3. C3 repair: fix any (m,k) violations introduced by SPS's demotions
     (Stage 1's pattern has zero slack, so any demotion breaks C3 for
     every window containing that job -- see repair_violations()).
  4. Heuristic loop:
       - sort tasks by DESCENDING unfairness (gamma_i), worst first
       - take the worst task; try its unscheduled jobs in natural order
         (job 1, 2, 3, ...)
       - for each job, try processors in order (0, 1, 2, ...), and
         within each processor try frequencies LOWEST first (cheapest
         energy first)
       - on the first feasible (job, processor, frequency): schedule it,
         recompute unfairness for every task, re-sort, and restart the
         loop (the new worst task may or may not be the same one)
       - if NO job of the current worst task is schedulable on ANY
         processor at ANY frequency: STOP THE ENTIRE HEURISTIC
         immediately (by design -- this version does not fall back to
         improving a different task)

IMPLEMENTATION NOTE on the feasibility check: the design as described
checks feasibility by computing the demand bound function (DBF) over a
job's [activation, deadline] interval against everything already
scheduled there. This implementation uses an EQUIVALENT but much more
efficient check instead -- direct preemptive EDF feasibility simulation
(same notion of feasibility, a standard result in real-time scheduling
theory, but O(n log n) per check instead of enumerating every candidate
interval). This matters in practice: the naive interval-enumeration
approach can blow up to out-of-memory on larger testcases (verified
directly on Baseline 1, where it went from crashing to running in 0.3s
after this same swap) since it mirrors the ILP's own exhaustive
constraint count, which can reach into the millions.

Self-contained: pattern generation, feasibility tracking, SPS, and C3
repair are all inlined in this single file (no imports from other
baseline/ILP folders), with only a local copy of preprocessing.py as a
dependency, consistent with the rest of this project's folder layout.

Usage: python heuristic1.py <testcase_filename>
       (filename resolved against preprocessing.TESTCASES_DIR)
"""

import sys
import heapq

from preprocessing import preprocess, TESTCASES_DIR


# =========================
# STAGE 1: E-PATTERN (evenly-distributed pattern generation)
# =========================

def generate_evenly_pattern(tasks, jobs):
    """
    Distance-based pattern: job j accepted iff floor(j*m/k) !=
    floor((j-1)*m/k). Satisfies (m,k) exactly for every sliding window.
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


# =========================
# SHARED FEASIBILITY STATE (EDF simulation + energy tracking)
# =========================

class SchedulerState:
    """
    Tracks, per processor, the set of committed jobs (arrival, deadline,
    execution time). Checks C2 (timing) feasibility via direct
    preemptive EDF simulation, and C4 (energy) via a running total --
    see module docstring for why EDF simulation is used instead of
    literal DBF-interval enumeration.
    """

    def __init__(self, tasks, jobs, energy_budget, freq_levels, n_prc, h_bar=1.0):
        self.tasks = tasks
        self.jobs = jobs
        self.energy_budget = energy_budget
        self.freq_levels = freq_levels
        self.n_prc = n_prc
        self.h_bar = h_bar

        self.total_energy = 0.0
        self.total_load = {x: 0.0 for x in range(n_prc)}
        self.committed = {x: [] for x in range(n_prc)}

        self.assignment = {}
        for i in range(len(tasks)):
            for j in jobs[i]:
                self.assignment[(i, j)] = None

    @staticmethod
    def _is_edf_feasible(job_list):
        """job_list: list of (arrival, deadline, exec_time)."""
        if not job_list:
            return True

        events = sorted(job_list, key=lambda jb: jb[0])
        n = len(events)
        heap = []
        idx = 0
        t = events[0][0]

        while idx < n or heap:
            while idx < n and events[idx][0] <= t + 1e-9:
                arrival, deadline, exec_time = events[idx]
                heapq.heappush(heap, (deadline, exec_time))
                idx += 1

            if not heap:
                t = events[idx][0]
                continue

            deadline, remaining = heapq.heappop(heap)
            next_arrival = events[idx][0] if idx < n else float("inf")
            time_slice = remaining if next_arrival == float("inf") else min(remaining, next_arrival - t)

            if time_slice >= remaining - 1e-9:
                finish_time = t + remaining
                if finish_time > deadline + 1e-9:
                    return False
                t = finish_time
            else:
                remaining -= time_slice
                t = next_arrival
                heapq.heappush(heap, (deadline, remaining))

        return True

    def can_place(self, i, j, x, f):
        e_i = self.tasks[i][0]
        p_i = self.tasks[i][1]
        f_y = self.freq_levels[f]
        exec_time = e_i / f_y

        job_energy = self.h_bar * (f_y ** 2) * e_i
        if self.total_energy + job_energy > self.energy_budget + 1e-9:
            return False

        arrival = (j - 1) * p_i
        deadline = j * p_i
        candidate_jobs = self.committed[x] + [(arrival, deadline, exec_time)]
        return self._is_edf_feasible(candidate_jobs)

    def place(self, i, j, x, f):
        e_i = self.tasks[i][0]
        p_i = self.tasks[i][1]
        f_y = self.freq_levels[f]
        exec_time = e_i / f_y

        arrival = (j - 1) * p_i
        deadline = j * p_i
        self.committed[x].append((arrival, deadline, exec_time))
        self.total_load[x] += exec_time

        self.total_energy += self.h_bar * (f_y ** 2) * e_i
        self.assignment[(i, j)] = (x, f)

    def try_place_best(self, i, j, freq_order=None):
        """Used by SPS and C3 repair: least-loaded processor first, given freq_order."""
        if freq_order is None:
            freq_order = range(len(self.freq_levels))

        proc_order = sorted(range(self.n_prc), key=lambda x: self.total_load[x])

        for x in proc_order:
            for f in freq_order:
                if self.can_place(i, j, x, f):
                    self.place(i, j, x, f)
                    return True
        return False


# =========================
# STAGE 2: SPS (Sum of Partial Solutions)
# =========================

def run_sps(tasks, jobs, pattern, energy_budget, freq_levels, n_prc, h_bar=1.0):
    state = SchedulerState(tasks, jobs, energy_budget, freq_levels, n_prc, h_bar)

    accepted_jobs = [
        (i, j) for i in range(len(tasks)) for j in jobs[i] if pattern[i][j] == 1
    ]
    accepted_jobs.sort(key=lambda ij: ij[1] * tasks[ij[0]][1])  # EDF order

    demoted = []
    for (i, j) in accepted_jobs:
        if not state.try_place_best(i, j):
            demoted.append((i, j))

    return state, demoted


# =========================
# STAGE 3: C3 REPAIR
# =========================

def find_violations(state, tasks, jobs, eta):
    """Returns list of (task_i, window_start, window_jobs) below m accepted."""
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
                continue

            skipped_in_window = [
                j for j in window_jobs if state.assignment[(i, j)] is None
            ]
            skipped_in_window.sort(key=lambda j: j * tasks[i][1])

            for j in skipped_in_window:
                if still_needed <= 0:
                    break
                if state.try_place_best(i, j, freq_order=freq_order):
                    still_needed -= 1
                    fixed_any = True

        if not fixed_any:
            return state, find_violations(state, tasks, jobs, eta)


# =========================
# STAGE 4: HEURISTIC LOOP (worst-task-first)
# =========================

def _gamma(state, jobs, i, eta_i):
    accepted = sum(1 for j in jobs[i] if state.assignment[(i, j)] is not None)
    return (eta_i - accepted) / eta_i


def run_heuristic(state, tasks, jobs, eta, freq_levels, n_prc):
    n_tsk = len(tasks)
    n_frq = len(freq_levels)
    iterations = 0
    stop_reason = None

    while True:
        gammas = [(_gamma(state, jobs, i, eta[i]), i) for i in range(n_tsk)]
        gammas.sort(reverse=True)
        worst_gamma, worst_i = gammas[0]

        if worst_gamma <= 0:
            stop_reason = "all tasks fully accepted"
            break

        unscheduled = [j for j in jobs[worst_i] if state.assignment[(worst_i, j)] is None]

        scheduled_something = False
        for j in unscheduled:
            for x in range(n_prc):
                for f in range(n_frq):  # lowest frequency first (freq_levels ascending)
                    if state.can_place(worst_i, j, x, f):
                        state.place(worst_i, j, x, f)
                        scheduled_something = True
                        break
                if scheduled_something:
                    break
            if scheduled_something:
                break

        if not scheduled_something:
            stop_reason = f"no job of task {worst_i} (worst, gamma={worst_gamma:.4f}) is schedulable"
            break

        iterations += 1

    return state, iterations, stop_reason


# =========================
# MAIN
# =========================

def run_heuristic1(testcase_filename):
    (
        n_prc, freq_levels, energy_budget, tasks,
        h_eff, eta, jobs, arrival_times, deadline_times, freq_indices,
    ) = preprocess(testcase_filename)

    pattern = generate_evenly_pattern(tasks, jobs)
    state, demoted = run_sps(tasks, jobs, pattern, energy_budget, freq_levels, n_prc)

    high_freq_first = list(range(len(freq_levels) - 1, -1, -1))
    state, unresolved = repair_violations(state, tasks, jobs, eta, freq_order=high_freq_first)

    state, iterations, stop_reason = run_heuristic(state, tasks, jobs, eta, freq_levels, n_prc)

    gammas = [_gamma(state, jobs, i, eta[i]) for i in range(len(tasks))]

    return {
        "testcase": testcase_filename,
        "phi": max(gammas),
        "total_energy": state.total_energy,
        "energy_budget": energy_budget,
        "gammas": gammas,
        "demoted_by_sps": len(demoted),
        "c3_unresolved": len(unresolved),
        "heuristic_iterations": iterations,
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "t0004.txt"

    result = run_heuristic1(filename)

    print(f"=== Heuristic 1: {result['testcase']} ===")
    print(f"Resolved path: {TESTCASES_DIR / filename}")
    print()
    for i, g in enumerate(result["gammas"]):
        print(f"  Task {i}: gamma = {g:.8f}")
    print()
    print(f"Phi (unfairness):        {result['phi']:.8f}")
    print(f"Total energy used:       {result['total_energy']:.8f} / "
          f"budget {result['energy_budget']}")
    print(f"Jobs demoted by SPS:     {result['demoted_by_sps']}")
    print(f"C3 violations unresolved: {result['c3_unresolved']}")
    print(f"Heuristic iterations:    {result['heuristic_iterations']}")
    print(f"Stopped because:         {result['stop_reason']}")
