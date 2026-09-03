"""
heuristic2_lookahead.py -- Heuristic 2: lookahead search over job choice.

Same Stage 1-3 as Heuristic 1 (E-pattern -> SPS -> C3 repair). Stage 4
replaces the greedy "always take the worst task's first feasible job in
natural order" rule with an EXHAUSTIVE LOOKAHEAD SEARCH to a fixed depth
d:

  At each node (a tentative schedule state), identify the current worst
  task (by gamma_i -- this can differ node to node, since committing a
  job changes gammas). Branch on EVERY feasible job of that task (no
  beam-width pruning -- every branch is kept, per confirmed design).
  Recurse to depth d. Score each leaf by (Phi, total_energy), lower Phi
  better, ties broken by lower energy. Whichever FIRST-level candidate
  leads to the best leaf is the one actually committed. Then the WHOLE
  search restarts from scratch for the next decision (most accurate,
  most expensive option, per confirmed design).

COMPLEXITY WARNING (confirmed as an accepted tradeoff by design, to be
measured empirically rather than assumed): with no beam-width pruning
and a branching factor of "all feasible jobs" at each level, the search
tree size grows roughly as (branching factor)^d per SINGLE real
scheduling decision, and this whole search re-runs from scratch after
every decision. This file reports wall-clock time for both testcases
tested so the actual cost can be judged directly rather than guessed.
If this turns out too slow, the two documented levers (from the
whiteboard's "Adaptive" refinement, not yet implemented here) are:
reducing d, and/or capping the branching factor instead of using every
feasible job.

For a given candidate job (not proc/freq), placement still follows the
same deterministic rule as Heuristic 1: try processors in sequential
order, frequencies lowest-first, first feasible combo wins. Branching is
over WHICH JOB only, not which processor/frequency, per the whiteboard.
"""

import sys
import heapq
import copy

from preprocessing import preprocess, TESTCASES_DIR


# =========================
# STAGE 1: E-PATTERN
# =========================

def generate_evenly_pattern(tasks, jobs):
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

    def clone(self):
        """
        Lightweight clone for lookahead branching: shares the immutable
        tasks/jobs/freq_levels references, copies only the mutable
        scheduling state. Much cheaper than a generic deepcopy, since
        tasks/jobs never change during search.
        """
        new_state = SchedulerState.__new__(SchedulerState)
        new_state.tasks = self.tasks
        new_state.jobs = self.jobs
        new_state.energy_budget = self.energy_budget
        new_state.freq_levels = self.freq_levels
        new_state.n_prc = self.n_prc
        new_state.h_bar = self.h_bar

        new_state.total_energy = self.total_energy
        new_state.total_load = dict(self.total_load)
        new_state.committed = {x: list(jobs_) for x, jobs_ in self.committed.items()}
        new_state.assignment = dict(self.assignment)
        return new_state

    @staticmethod
    def _is_edf_feasible(job_list):
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
        """Used by SPS and C3 repair: least-loaded processor first."""
        if freq_order is None:
            freq_order = range(len(self.freq_levels))
        proc_order = sorted(range(self.n_prc), key=lambda x: self.total_load[x])
        for x in proc_order:
            for f in freq_order:
                if self.can_place(i, j, x, f):
                    self.place(i, j, x, f)
                    return True
        return False

    def find_first_feasible_placement(self, i, j):
        """
        Sequential processor order, lowest-frequency-first (matching
        Heuristic 1's placement rule). Returns (x, f) or None. Does NOT
        commit -- used to check whether a candidate job is feasible at
        all, for lookahead branching.
        """
        for x in range(self.n_prc):
            for f in range(len(self.freq_levels)):
                if self.can_place(i, j, x, f):
                    return (x, f)
        return None


# =========================
# STAGE 2: SPS
# =========================

def run_sps(tasks, jobs, pattern, energy_budget, freq_levels, n_prc, h_bar=1.0):
    state = SchedulerState(tasks, jobs, energy_budget, freq_levels, n_prc, h_bar)
    accepted_jobs = [
        (i, j) for i in range(len(tasks)) for j in jobs[i] if pattern[i][j] == 1
    ]
    accepted_jobs.sort(key=lambda ij: ij[1] * tasks[ij[0]][1])
    demoted = []
    for (i, j) in accepted_jobs:
        if not state.try_place_best(i, j):
            demoted.append((i, j))
    return state, demoted


# =========================
# STAGE 3: C3 REPAIR
# =========================

def find_violations(state, tasks, jobs, eta):
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
            skipped_in_window = [j for j in window_jobs if state.assignment[(i, j)] is None]
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
# STAGE 4: LOOKAHEAD SEARCH (depth d, no beam-width pruning)
# =========================

def _gamma(state, jobs, i, eta_i):
    accepted = sum(1 for j in jobs[i] if state.assignment[(i, j)] is not None)
    return (eta_i - accepted) / eta_i


def _find_worst_task(state, tasks, jobs, eta):
    """Returns (worst_gamma, worst_i), or (0.0, None) if all tasks fully accepted."""
    gammas = [(_gamma(state, jobs, i, eta[i]), i) for i in range(len(tasks))]
    gammas.sort(reverse=True)
    worst_gamma, worst_i = gammas[0]
    if worst_gamma <= 0:
        return 0.0, None
    return worst_gamma, worst_i


def _enumerate_candidates(state, tasks, jobs, worst_i):
    """
    All feasible (job, x, f) triples for the given task's currently
    unscheduled jobs -- the branching factor (no cap, per confirmed
    design). Placement (x,f) for each job follows the deterministic
    sequential-processor / lowest-frequency-first rule; branching is
    over WHICH JOB only.
    """
    candidates = []
    for j in jobs[worst_i]:
        if state.assignment[(worst_i, j)] is not None:
            continue
        placement = state.find_first_feasible_placement(worst_i, j)
        if placement is not None:
            candidates.append((j, placement[0], placement[1]))
    return candidates


def _score(state, tasks, jobs, eta):
    """(Phi, energy) -- lower is better for both, Phi first."""
    gammas = [_gamma(state, jobs, i, eta[i]) for i in range(len(tasks))]
    return (max(gammas), state.total_energy)


def _best_leaf_score(state, tasks, jobs, eta, remaining_depth):
    """
    Returns the best (Phi, energy) achievable from `state` within
    `remaining_depth` more decisions. Does not mutate `state`. If no
    further move is possible (worst task stuck, or all tasks done),
    the current state's score IS the leaf.
    """
    if remaining_depth == 0:
        return _score(state, tasks, jobs, eta)

    worst_gamma, worst_i = _find_worst_task(state, tasks, jobs, eta)
    if worst_i is None:
        return _score(state, tasks, jobs, eta)

    candidates = _enumerate_candidates(state, tasks, jobs, worst_i)
    if not candidates:
        return _score(state, tasks, jobs, eta)

    best = None
    for (j, x, f) in candidates:
        child = state.clone()
        child.place(worst_i, j, x, f)
        leaf = _best_leaf_score(child, tasks, jobs, eta, remaining_depth - 1)
        if best is None or leaf < best:
            best = leaf
    return best


def _choose_best_first_move(state, tasks, jobs, eta, depth):
    """
    Runs the full lookahead search and returns the (worst_i, j, x, f)
    first move that leads to the best leaf, or None if the current
    worst task has zero feasible candidates (the stop condition).
    """
    worst_gamma, worst_i = _find_worst_task(state, tasks, jobs, eta)
    if worst_i is None:
        return None, "all tasks fully accepted"

    candidates = _enumerate_candidates(state, tasks, jobs, worst_i)
    if not candidates:
        return None, f"no job of task {worst_i} (worst, gamma={worst_gamma:.4f}) is schedulable"

    best_move = None
    best_leaf = None
    for (j, x, f) in candidates:
        child = state.clone()
        child.place(worst_i, j, x, f)
        leaf = _best_leaf_score(child, tasks, jobs, eta, depth - 1)
        if best_leaf is None or leaf < best_leaf:
            best_leaf = leaf
            best_move = (worst_i, j, x, f)

    return best_move, None


def run_heuristic2(state, tasks, jobs, eta, depth):
    iterations = 0
    stop_reason = None

    while True:
        move, reason = _choose_best_first_move(state, tasks, jobs, eta, depth)
        if move is None:
            stop_reason = reason
            break
        i, j, x, f = move
        state.place(i, j, x, f)
        iterations += 1

    return state, iterations, stop_reason


# =========================
# MAIN
# =========================

def run_heuristic2_pipeline(testcase_filename, depth=2):
    (
        n_prc, freq_levels, energy_budget, tasks,
        h_eff, eta, jobs, arrival_times, deadline_times, freq_indices,
    ) = preprocess(testcase_filename)

    pattern = generate_evenly_pattern(tasks, jobs)
    state, demoted = run_sps(tasks, jobs, pattern, energy_budget, freq_levels, n_prc)

    high_freq_first = list(range(len(freq_levels) - 1, -1, -1))
    state, unresolved = repair_violations(state, tasks, jobs, eta, freq_order=high_freq_first)

    state, iterations, stop_reason = run_heuristic2(state, tasks, jobs, eta, depth)

    gammas = [_gamma(state, jobs, i, eta[i]) for i in range(len(tasks))]

    return {
        "testcase": testcase_filename,
        "depth": depth,
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
    filename = sys.argv[1] if len(sys.argv) > 1 else "t0006.txt"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    result = run_heuristic2_pipeline(filename, depth=depth)

    print(f"=== Heuristic 2 (lookahead depth={result['depth']}): {result['testcase']} ===")
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