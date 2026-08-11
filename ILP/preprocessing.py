"""
Preprocessing for the EPFS (Energy & Fairness aware weakly-hard real-time
scheduling) problem.

Reads a plain-text testcase file and derives every quantity the ILP
(model.py) and the heuristic pipeline need: the effective hyperperiod
h_eff, per-task job counts (eta), job index lists, and the DBF interval
endpoints (arrival/deadline timestamps).

Expected file format (produced by testcase_generator.py):

    <n_prc>
    <freq_level_1> <freq_level_2> ... <freq_level_nfrq>
    <energy_budget>
    <e_1> <p_1> <m_1> <k_1>
    <e_2> <p_2> <m_2> <k_2>
    ...

All testcase files live in a shared `testcases/` folder that is a SIBLING
of this file's parent directory, i.e.:

    <project_root>/
        <this_folder>/preprocessing.py   (and model.py, testcase_generator.py)
        testcases/<filename>.txt

TESTCASES_DIR below is the single source of truth for that location.
testcase_generator.py imports it directly (rather than re-deriving its
own copy), so a file it writes is always immediately readable here under
the same name -- no manual copying, no path drift between scripts.
"""

import math
from pathlib import Path


# =========================
# SHARED PATH (single source of truth -- import this elsewhere, don't
# re-derive it)
# =========================

TESTCASES_DIR = Path(__file__).resolve().parent.parent / "testcases/batch_v1"


# =========================
# READ TEST CASE
# =========================

def read_test_case(filename):
    """
    filename: just the file's name (e.g. "t1.txt"), not a path --
    always resolved against TESTCASES_DIR.
    """
    filepath = TESTCASES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Testcase '{filename}' not found at resolved path: {filepath}\n"
            f"Make sure it exists in {TESTCASES_DIR}, or that "
            f"testcase_generator.py has been run to create it."
        )

    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_prc = int(lines[0])
    freq_levels = list(map(float, lines[1].split()))
    energy_budget = float(lines[2])

    tasks = []
    for line in lines[3:]:
        e, p, m, k = map(int, line.split())
        tasks.append((e, p, m, k))

    return n_prc, freq_levels, energy_budget, tasks


# =========================
# LCM HELPERS
# =========================

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def lcm_list(values):
    result = values[0]
    for v in values[1:]:
        result = lcm(result, v)
    return result


# =========================
# COMPUTE h_eff
# =========================

def compute_h_eff(tasks):
    """
    h_eff = lcm(k_i * p_i) over all tasks -- the effective hyperperiod
    over which both timing AND (m,k) constraints must be verified
    (paper Section I.A). This is always an exact multiple of every p_i
    AND every k_i, so eta_i = h_eff // p_i below never truncates, and
    eta_i is itself always a multiple of k_i (eta_i >= k_i always holds,
    so C3's sliding-window construction in model.py never operates on a
    task with fewer jobs than its own window size).
    """
    return lcm_list([p * k for e, p, m, k in tasks])


# =========================
# COMPUTE ETA
# =========================

def compute_eta(tasks, h_eff):
    return [h_eff // p for e, p, m, k in tasks]


# =========================
# GENERATE JOBS
# =========================

def generate_jobs(eta):
    return {i: list(range(1, eta[i] + 1)) for i in range(len(eta))}


# =========================
# GENERATE TIMES
# =========================

def generate_times(tasks, jobs):
    """
    Distinct arrival and deadline timestamps across all jobs of all
    tasks -- these form the candidate DBF interval endpoints [t1, t2]
    used by model.py's C2 constraint (Eq. 7-8).
    """
    arrivals = set()
    deadlines = set()

    for i in range(len(tasks)):
        _, p, _, _ = tasks[i]
        for j in jobs[i]:
            arrivals.add((j - 1) * p)
            deadlines.add(j * p)

    return sorted(arrivals), sorted(deadlines)


# =========================
# PREPROCESS
# =========================

def preprocess(filename):
    n_prc, freq_levels, energy_budget, tasks = read_test_case(filename)

    h_eff = compute_h_eff(tasks)
    eta = compute_eta(tasks, h_eff)
    jobs = generate_jobs(eta)
    arrivals, deadlines = generate_times(tasks, jobs)
    freq_indices = list(range(len(freq_levels)))

    return (
        n_prc,
        freq_levels,
        energy_budget,
        tasks,
        h_eff,
        eta,
        jobs,
        arrivals,
        deadlines,
        freq_indices,
    )


# =========================
# TEST
# =========================

if __name__ == "__main__":
    (
        n_prc, freq_levels, energy_budget, tasks,
        h_eff, eta, jobs, arrivals, deadlines, freq_indices,
    ) = preprocess("t1.txt")

    print("Resolved testcases folder:", TESTCASES_DIR)
    print("Processors:", n_prc)
    print("Frequency Levels:", freq_levels)
    print("Energy Budget:", energy_budget)
    print("Tasks (e,p,m,k):", tasks)
    print("h_eff:", h_eff)
    print("eta:", eta)
    print("Total jobs:", sum(eta))
    print("Distinct arrivals:", len(arrivals), "| deadlines:", len(deadlines))
    print("freq indices:", freq_indices)