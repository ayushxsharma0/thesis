"""
EPFS ILP baseline (exact optimum via Gurobi).

Implements the ILP formulation exactly:
  C1 - allocation constraint          (Eq. 5-6)
  C2 - DBF timing constraint           (Eq. 7-8)
  C3 - weakly-hard (m,k) constraint    (Eq. 9, sliding-window form)
  C4 - energy budget constraint        (Eq. 1-2, 10)
  Objective - min-max fairness         (Eq. 4, 11-12)
"""

import gurobipy as gp
from gurobipy import GRB

from preprocessing import preprocess, TESTCASES_DIR

 
# =========================
# CONFIG
# =========================
# The only line you should need to touch to switch testcases. Keep this
# in sync with testcase_generator.py's TESTCASE_FILENAME if you're
# regenerating -- both resolve against the exact same TESTCASES_DIR
# (imported from preprocessing.py), so as long as the filename string
# matches, the two scripts can never end up looking at different files.

TESTCASE_FILENAME = "t0004.txt"
TIME_LIMIT_SECONDS = None   # e.g. 300 to cap solve time; None = no limit


# =========================
# PREPROCESS
# =========================

(
    n_prc,
    freq_levels,
    energy_budget,
    tasks,
    h_eff,
    eta,
    jobs,
    arrival_times,
    deadline_times,
    freq_indices,
) = preprocess(TESTCASE_FILENAME)

h_bar = 1.0

print(f"Loaded testcase: {TESTCASES_DIR / TESTCASE_FILENAME}")
print(f"  n_prc={n_prc}, n_frq={len(freq_levels)}, n_tsk={len(tasks)}, "
      f"h_eff={h_eff}, total jobs={sum(eta)}, energy_budget={energy_budget}")


# =========================
# MODEL
# =========================

model = gp.Model("EPFS_ILP")

if TIME_LIMIT_SECONDS is not None:
    model.Params.TimeLimit = TIME_LIMIT_SECONDS


# =========================
# DECISION VARIABLES
# =========================

X = {
    (i, j, x, f): model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{x}_{f}")
    for i in range(len(tasks))
    for j in jobs[i]
    for x in range(n_prc)
    for f in freq_indices
}

Y = {
    (i, j): model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}")
    for i in range(len(tasks))
    for j in jobs[i]
}


# =========================
# C1: allocation constraint (Eq. 5-6)
# =========================

for i in range(len(tasks)):
    for j in jobs[i]:
        job_alloc = gp.quicksum(
            X[i, j, x, f] for x in range(n_prc) for f in freq_indices
        )
        model.addConstr(job_alloc <= 1, name=f"C1_alloc_{i}_{j}")
        model.addConstr(Y[i, j] == job_alloc, name=f"C1_Y_{i}_{j}")


# =========================
# C2: DBF timing constraint (Eq. 7-8)
# =========================

time_pairs = [
    (t1, t2) for t1 in arrival_times for t2 in deadline_times if t2 > t1
]

# relevant_jobs for a given (t1,t2) does not depend on the processor x,
# so precompute it once per interval instead of recomputing it n_prc
# times inside the processor loop below.
relevant_jobs_by_interval = {}
for t1, t2 in time_pairs:
    relevant_jobs = []
    for i in range(len(tasks)):
        p = tasks[i][1]
        for j in jobs[i]:
            arrival = (j - 1) * p
            deadline = j * p
            if arrival >= t1 and deadline <= t2:
                relevant_jobs.append((i, j))
    relevant_jobs_by_interval[(t1, t2)] = relevant_jobs

for x in range(n_prc):
    for (t1, t2), relevant_jobs in relevant_jobs_by_interval.items():
        if not relevant_jobs:
            continue
        demand = gp.quicksum(
            (tasks[i][0] / freq_levels[f]) * X[i, j, x, f]
            for i, j in relevant_jobs
            for f in freq_indices
        )
        model.addConstr(demand <= t2 - t1, name=f"C2_{x}_{t1}_{t2}")


# =========================
# C3: weakly-hard (m,k) constraint, sliding window (Eq. 9, strict form)
# =========================
# eta[i] is always >= k (guaranteed by h_eff being a multiple of k*p, see
# preprocessing.compute_h_eff), so this range is always non-empty.

for i in range(len(tasks)):
    e, p, m, k = tasks[i]
    for start in range(1, eta[i] - k + 2):
        model.addConstr(
            gp.quicksum(Y[i, j] for j in range(start, start + k)) >= m,
            name=f"C3_{i}_{start}",
        )


# =========================
# C4: energy budget constraint (Eq. 1-2, 10)
# =========================

total_energy = gp.quicksum(
    h_bar * (freq_levels[f] ** 2) * tasks[i][0] * X[i, j, x, f]
    for i in range(len(tasks))
    for j in jobs[i]
    for x in range(n_prc)
    for f in freq_indices
)

model.addConstr(total_energy <= energy_budget, name="C4_energy")


# =========================
# OBJECTIVE: min-max fairness (Eq. 4, 11-12)
# =========================

Phi = model.addVar(vtype=GRB.CONTINUOUS, name="Phi")

for i in range(len(tasks)):
    accepted = gp.quicksum(Y[i, j] for j in jobs[i])
    missed = eta[i] - accepted
    gamma_i = missed / eta[i]
    model.addConstr(Phi >= gamma_i, name=f"Obj_gamma_{i}")

model.setObjective(Phi, GRB.MINIMIZE)


# =========================
# SOLVE
# =========================

print(f"\nModel size: {model.NumVars} vars, {model.NumConstrs} constraints")

model.optimize()

print("\nStatus code:", model.status)

if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
elif model.status == GRB.INFEASIBLE:
    print("Model is INFEASIBLE.")
    model.computeIIS()
    model.write("model.ilp")
    print("Irreducible Inconsistent Subsystem written to model.ilp "
          "— check it to see conflicting constraints.")
elif model.status == GRB.UNBOUNDED:
    print("Model is UNBOUNDED.")
elif model.status == GRB.INF_OR_UNBD:
    print("Model is infeasible or unbounded.")
else:
    print(f"Optimization ended with status {model.status}")


# =========================
# OUTPUT
# =========================
# Guarded by SolCount (not strictly GRB.OPTIMAL), so a feasible-but-not-
# proven-optimal result (e.g. hit TIME_LIMIT_SECONDS) is still reported
# instead of crashing; only a genuinely infeasible/unsolved model skips
# this section entirely.

if model.SolCount > 0:
    print("\n=== Solution ===")
    for i in range(len(tasks)):
        for j in jobs[i]:
            assigned = False
            for x in range(n_prc):
                for f in freq_indices:
                    if X[i, j, x, f].X > 0.5:
                        print(f"Task {i}, Job {j} -> Processor {x} "
                              f"@ Frequency {freq_levels[f]}")
                        assigned = True
            if not assigned:
                print(f"Task {i}, Job {j} -> SKIPPED")

    print("\nPhi:", Phi.X)
    print("Total Energy:", total_energy.getValue())

                   
    ilp_phi = Phi.X
    ilp_patterns = {
        i: [round(Y[i, j].X) for j in jobs[i]]
        for i in range(len(tasks))
    }
else:
    print("\nNo solution available to report "
          "(model was infeasible, unbounded, or not solved).")
    ilp_phi = None
    ilp_patterns = {}