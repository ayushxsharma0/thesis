# =========================
# IMPORTS
# =========================

import math
from pathlib import Path

# =========================
# READ TEST CASE
# =========================

def read_test_case(filename):

    filepath = Path(__file__).parent.parent / "testcases" / filename

    with open(filepath, "r") as f:

        lines = [

            line.strip()

            for line in f

            if line.strip()

        ]

    n_prc = int(lines[0])

    freq_levels = list(
        map(float, lines[1].split())
    )

    energy_budget = float(lines[2])

    tasks = []

    for line in lines[3:]:

        e, p, m, k = map(
            int,
            line.split()
        )

        tasks.append(
            (e, p, m, k)
        )

    return (
        n_prc,
        freq_levels,
        energy_budget,
        tasks
    )


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

    return lcm_list(

        [p * k for e, p, m, k in tasks]

    )


# =========================
# COMPUTE ETA
# =========================

def compute_eta(tasks, h_eff):

    return [

        h_eff // p

        for e, p, m, k in tasks

    ]


# =========================
# GENERATE JOBS
# =========================

def generate_jobs(eta):

    return {

        i: list(range(1, eta[i] + 1))

        for i in range(len(eta))

    }


# =========================
# GENERATE TIMES
# =========================

def generate_times(tasks, jobs):

    arrivals = set()
    deadlines = set()

    for i in range(len(tasks)):

        _, p, _, _ = tasks[i]

        for j in jobs[i]:

            arrivals.add(
                (j - 1) * p
            )

            deadlines.add(
                j * p
            )

    return (
        sorted(arrivals),
        sorted(deadlines)
    )


# =========================
# PREPROCESS
# =========================

def preprocess(filename="../testcase/t5.txt"):

    (
        n_prc,
        freq_levels,
        energy_budget,
        tasks

    ) = read_test_case(filename)

    # temporarily keep fixed if needed
    # h_eff = 65

    h_eff = compute_h_eff(tasks)

    eta = compute_eta(
        tasks,
        h_eff
    )

    jobs = generate_jobs(
        eta
    )

    arrivals, deadlines = generate_times(
        tasks,
        jobs
    )

    freq_indices = list(
        range(
            len(freq_levels)
        )
    )

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

        freq_indices

    )


# =========================
# TEST
# =========================

if __name__ == "__main__":

    (
        n_prc,
        freq_levels,
        energy_budget,
        tasks,

        h_eff,
        eta,
        jobs,

        arrivals,
        deadlines,

        freq_indices

    ) = preprocess("testcase.txt")

    print("Processors:", n_prc)

    print("Frequency Levels:", freq_levels)

    print("Energy Budget:", energy_budget)

    print("Tasks:", tasks)

    print("h_eff:", h_eff)

    print("eta:", eta)

    print("jobs:", jobs)

    print("arrivals:", arrivals)

    print("deadlines:", deadlines)

    print("freq indices:", freq_indices)