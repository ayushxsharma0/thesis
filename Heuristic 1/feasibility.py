"""
Shared feasibility-tracking state used by SPS (Stage 2) and Fairness (Stage 3).

Tracks, incrementally, exactly the same quantities the ILP constrains:
  - C2 / DBF demand per processor per (t1, t2) interval   (Eq. 7, Eq. 8)
  - C4 / total system energy                               (Eq. 1-2, Eq. 10)

so that Stage 2 and Stage 3 can both ask "would job (i,j) fit on
processor x at frequency f without breaking C2 or C4?" in O(#intervals
containing that job) time, instead of recomputing full sums from scratch
on every candidate placement.

C3 ((m,k)) is NOT tracked here -- it's handled structurally: Stage 1
(pattern.py) only ever proposes patterns that already satisfy C3, and
Stage 2/3 only ever ADD accepted jobs on top of that, never remove them.
So a task's accepted-job count only ever goes up relative to Stage 1's
baseline, meaning C3 remains satisfied by construction throughout.
"""


class SchedulerState:

    def __init__(self, tasks, jobs, time_pairs, energy_budget, freq_levels,
                 n_prc, h_bar=1.0):
        self.tasks = tasks              # list of (e, p, m, k)
        self.jobs = jobs                # dict i -> [1..eta_i]
        self.time_pairs = time_pairs    # list of (t1, t2) candidate DBF intervals
        self.energy_budget = energy_budget
        self.freq_levels = freq_levels
        self.n_prc = n_prc
        self.h_bar = h_bar

        self.total_energy = 0.0

        # demand[x][(t1, t2)] -> float, current committed demand on processor x
        self.demand = {
            x: {tp: 0.0 for tp in time_pairs} for x in range(n_prc)
        }

        # assignment[(i, j)] -> (x, f) if accepted & placed, else None
        self.assignment = {}
        for i in range(len(tasks)):
            for j in jobs[i]:
                self.assignment[(i, j)] = None

        # precompute, for each job, which DBF intervals it falls fully inside
        self.job_intervals = self._precompute_job_intervals()

    def _precompute_job_intervals(self):
        job_intervals = {}
        for i, (e, p, m, k) in enumerate(self.tasks):
            for j in self.jobs[i]:
                arrival = (j - 1) * p
                deadline = j * p
                job_intervals[(i, j)] = [
                    tp for tp in self.time_pairs
                    if arrival >= tp[0] and deadline <= tp[1]
                ]
        return job_intervals

    def can_place(self, i, j, x, f):
        """Check C2 (DBF) and C4 (energy) WITHOUT committing the change."""
        e_i = self.tasks[i][0]
        f_y = self.freq_levels[f]
        exec_time = e_i / f_y

        # C2: DBF feasibility on processor x over every interval containing (i,j)
        for tp in self.job_intervals[(i, j)]:
            t1, t2 = tp
            if self.demand[x][tp] + exec_time > (t2 - t1) + 1e-9:
                return False

        # C4: energy budget
        job_energy = self.h_bar * (f_y ** 2) * e_i
        if self.total_energy + job_energy > self.energy_budget + 1e-9:
            return False

        return True

    def place(self, i, j, x, f):
        """Commit job (i,j) onto processor x at frequency index f."""
        e_i = self.tasks[i][0]
        f_y = self.freq_levels[f]
        exec_time = e_i / f_y

        for tp in self.job_intervals[(i, j)]:
            self.demand[x][tp] += exec_time

        self.total_energy += self.h_bar * (f_y ** 2) * e_i
        self.assignment[(i, j)] = (x, f)

    def remove(self, i, j):
        """Undo a placement (kept for future backtracking heuristics)."""
        placement = self.assignment[(i, j)]
        if placement is None:
            return
        x, f = placement
        e_i = self.tasks[i][0]
        f_y = self.freq_levels[f]
        exec_time = e_i / f_y

        for tp in self.job_intervals[(i, j)]:
            self.demand[x][tp] -= exec_time

        self.total_energy -= self.h_bar * (f_y ** 2) * e_i
        self.assignment[(i, j)] = None

    def try_place_best(self, i, j, freq_order=None):
        """
        Try to place job (i,j) on any (processor, frequency) combination.

        Processors are tried least-loaded first (most likely to have DBF
        slack). Frequencies default to lowest-first (cheapest energy).
        Commits and returns True on the first feasible slot found;
        returns False (no state change) if nothing works.
        """
        if freq_order is None:
            freq_order = range(len(self.freq_levels))

        proc_order = sorted(range(self.n_prc), key=self._load)

        for x in proc_order:
            for f in freq_order:
                if self.can_place(i, j, x, f):
                    self.place(i, j, x, f)
                    return True
        return False

    def _load(self, x):
        """Rough load estimate: total demand committed on processor x so far."""
        return sum(self.demand[x].values())
