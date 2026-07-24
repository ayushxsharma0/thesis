"""
Automated testcase generator for the EPFS problem.

Grounded in standard real-time scheduling testcase generation practice:

  - Utilization generation: UUniFast-Discard (Bini & Buttazzo's UUniFast,
    with the "discard if any u_i > 1" extension needed for multiprocessor
    settings, since no single task may need more than one full processor
    -- this is a modeling requirement, not a license limitation, so it
    stays regardless of solver license).

  - Period generation: adapted from the standard log-uniform period
    approach (Davis et al.) into a STRICT geometric/harmonic version,
    since this paper's system model requires p_{i+1}/p_i = p_{i+2}/p_{i+1}
    for all i -- i.e. constant ratio between consecutive periods. This
    remains the one hard structural requirement on generated testcases.

  - (m,k) generation: tolerance-banded random selection, following
    "Global Scheduling of Weakly-Hard Real-Time Tasks using Job-Level
    Priority Classes" (m_i chosen randomly among values satisfying a
    desired m_i/k_i band -- e.g. low-tolerance tasks get m close to k,
    high-tolerance tasks get m small relative to k).

  - Energy budget: set as a tunable fraction between the mandatory-only
    baseline and full-acceptance energy, each priced at every task's OWN
    minimum feasible frequency (not a shared global minimum -- a
    high-utilization task may be physically barred from cheap frequency
    levels, and pricing it as if it could use them produces a budget
    that is infeasible by construction). Gives a genuine fairness/energy
    tradeoff instead of a trivial all-accept or all-skip regime.

  - Model-size reporting: estimates the ILP's exact variable/constraint
    counts (mirroring model.py's own construction) purely for
    visibility -- with an academic Gurobi license there is no hard cap
    to enforce, but h_eff still grows EXPONENTIALLY with n_tsk under
    strict harmonic periods, so a heads-up before launching a very large
    solve is still useful.

All testcases are written into the SAME folder preprocessing.py reads
from (TESTCASES_DIR, imported directly -- not re-derived), so the
generator and the solver can never disagree about where a file lives.
"""

import math
import random

from preprocessing import TESTCASES_DIR


# =========================
# UUNIFAST-DISCARD
# =========================

def uunifast_discard(n, U, u_max=1.0, max_attempts=1000):
    """
    Generate n utilization values summing to U, each in (0, u_max].
    Retries (discards) whenever any generated value exceeds u_max --
    required for multiprocessor settings where U may exceed 1 but no
    single task may need more than one full processor.
    """
    for _ in range(max_attempts):
        utils = []
        sum_u = U
        ok = True
        for i in range(1, n):
            next_sum_u = sum_u * (random.random() ** (1.0 / (n - i)))
            u_i = sum_u - next_sum_u
            if u_i > u_max:
                ok = False
                break
            utils.append(u_i)
            sum_u = next_sum_u
        if ok and sum_u <= u_max:
            utils.append(sum_u)
            return utils
    raise RuntimeError(
        f"UUniFast-Discard failed to generate a valid utilization vector "
        f"after {max_attempts} attempts (U={U}, n={n}, u_max={u_max}). "
        f"Try a lower total utilization U or more tasks n."
    )


# =========================
# HARMONIC PERIOD GENERATION
# =========================

def generate_harmonic_periods(n_tsk, p_min=4, ratio=None, ratio_choices=(2, 3, 5, 6)):
    """
    Generate n_tsk periods in STRICT geometric progression:
        p_i = p_min * ratio^(i-1)
    This satisfies p_{i+1}/p_i = p_{i+2}/p_{i+1} = ratio for every i,
    matching the paper's harmonic condition exactly (not just pairwise
    divisibility) -- this is the one hard requirement on every generated
    testcase.

    NOTE: h_eff grows roughly as 2 * p_max = 2 * p_min * ratio^(n_tsk-1),
    which is EXPONENTIAL in n_tsk. This is a PRACTICALITY note, not a
    license one -- even an unlimited Gurobi license takes real time to
    solve a model with hundreds of thousands of jobs. Keep ratio small
    (2 or 3) and n_tsk modest unless you deliberately want a
    stress-test-sized instance.
    """
    r = ratio if ratio is not None else random.choice(ratio_choices)
    periods = [p_min * (r ** i) for i in range(n_tsk)]
    return periods, r


# =========================
# (m, k) GENERATION
# =========================

def generate_mk(n_tsk, k_range=(2, 4), tolerance="mixed"):
    """
    Generate (m_i, k_i) pairs, tolerance-banded per Global Scheduling of
    Weakly-Hard Real-Time Tasks using Job-Level Priority Classes:
    m_i is chosen randomly among the values satisfying a desired m_i/k_i
    band for the task's k_i.

    tolerance:
      "low"   -> strict tasks: m in the upper half of [1, k-1]
                 (few misses tolerated)
      "high"  -> tolerant tasks: m in the lower half of [1, k-1]
                 (many misses tolerated)
      "mixed" -> m drawn uniformly from all of [1, k-1] (random per task)
    """
    mk_list = []
    for _ in range(n_tsk):
        k = random.randint(*k_range)
        if k < 2:
            k = 2  # k=1 gives a trivial hard (no-skip) task; avoid by default

        if tolerance == "low":
            lo = max(1, (k // 2) + 1)
            hi = k - 1
        elif tolerance == "high":
            lo = 1
            hi = max(1, k // 2)
        else:  # mixed
            lo, hi = 1, k - 1

        m = random.randint(lo, hi) if hi >= lo else 1
        mk_list.append((m, k))
    return mk_list


# =========================
# LCM HELPERS (pure math -- independent of preprocessing.py's path logic)
# =========================

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def lcm_list(values):
    result = values[0]
    for v in values[1:]:
        result = lcm(result, v)
    return result


# =========================
# MODEL SIZE ESTIMATION (mirrors model.py's exact constraint construction)
# =========================

def estimate_model_size(tasks, n_prc, n_frq, h_eff):
    """
    Computes the EXACT variable/constraint counts model.py would build
    for this testcase. Purely informational now (no license cap to
    enforce) -- useful to know how large a solve you're about to launch.
    """
    eta = [h_eff // p for (e, p, m, k) in tasks]
    total_jobs = sum(eta)

    n_vars = total_jobs * n_prc * n_frq + total_jobs  # X + Y

    c1 = 2 * total_jobs  # <=1 sum, and Y equality, per job

    c3 = 0  # one constraint per sliding window per task
    for i, (e, p, m, k) in enumerate(tasks):
        n = eta[i]
        if n >= k:
            c3 += (n - k + 1)

    arrivals = set()
    deadlines = set()
    for i, (e, p, m, k) in enumerate(tasks):
        for j in range(1, eta[i] + 1):
            arrivals.add((j - 1) * p)
            deadlines.add(j * p)
    time_pairs = [
        (t1, t2) for t1 in sorted(arrivals) for t2 in sorted(deadlines) if t2 > t1
    ]
    c2 = n_prc * len(time_pairs)

    c4 = 1

    n_constrs = c1 + c2 + c3 + c4

    return {
        "eta": eta,
        "total_jobs": total_jobs,
        "n_vars": n_vars,
        "n_constrs": n_constrs,
        "breakdown": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
    }


# =========================
# FREQUENCY LEVELS
# =========================

def generate_frequency_levels(n_frq, f_min=0.4, style="uniform"):
    """
    style="uniform": n_frq evenly spaced levels in [f_min, 1.0]
    style="xscale":  normalized levels resembling a real DVFS processor
                     (Intel XScale-style discrete steps), truncated/padded
                     to n_frq levels if needed. Always guarantees 1.0 is
                     included -- omitting it could leave a high-
                     utilization task with NO feasible frequency at all,
                     an infeasibility no energy budget could ever fix.
    """
    if style == "xscale":
        base = [0.15, 0.4, 0.6, 0.8, 1.0]
        if n_frq >= len(base):
            levels = list(base)
            while len(levels) < n_frq:
                levels.append(round(random.uniform(0.15, 1.0), 2))
            return sorted(set(levels))[:n_frq]
        if n_frq == 1:
            return [1.0]
        sampled = random.sample(base[:-1], n_frq - 1)
        return sorted(sampled + [1.0])

    if n_frq == 1:
        return [1.0]
    step = (1.0 - f_min) / (n_frq - 1)
    return [round(f_min + i * step, 3) for i in range(n_frq)]


# =========================
# PER-TASK MINIMUM FEASIBLE FREQUENCY
# =========================

def min_feasible_freq(e_i, p_i, freq_levels):
    """
    Returns the smallest frequency level f_y such that e_i / f_y <= p_i,
    i.e. the cheapest frequency at which this job can meet its OWN
    deadline in isolation. A job with high utilization (e_i/p_i close to
    1) may be barred from every "cheap" frequency level -- using a
    single global minimum frequency for cost estimation ignores this and
    can produce an energy budget that is infeasible by construction.
    """
    u_i = e_i / p_i
    feasible = [f for f in freq_levels if f >= u_i - 1e-9]
    if not feasible:
        raise ValueError(
            f"Task with e={e_i}, p={p_i} has utilization {u_i:.3f}, which "
            f"exceeds every available frequency level {freq_levels}. "
            f"This task cannot meet its own deadline even in isolation -- "
            f"raise the max frequency level or lower this task's utilization."
        )
    return min(feasible)


# =========================
# ENERGY BUDGET
# =========================

def compute_energy_budget(tasks, eta, freq_levels, slack_fraction=0.4, h_bar=1.0):
    """
    Sets energy_budget = mandatory_energy + slack_fraction * (full_energy - mandatory_energy),
    where mandatory_energy and full_energy are evaluated at EACH TASK'S
    OWN minimum feasible frequency (not a shared global minimum).

      - mandatory_energy: cost of accepting exactly m out of every k jobs
                          per task (Stage 1's baseline pattern) -- this
                          is a provable LOWER BOUND on required energy,
                          since the distance-based pattern achieves the
                          minimum possible mandatory job count for C3.
      - full_energy:      cost of accepting every job of every task

    slack_fraction=0.0 -> budget = mandatory baseline (tight, little room for fairness gains)
    slack_fraction=1.0 -> budget = full acceptance (energy never binds)
    Default 0.4 leaves genuine tradeoff room.

    NOTE: this still only prices each job at ITS OWN minimum feasible
    frequency in ISOLATION. Real DBF/processor congestion can still
    force some jobs onto a higher frequency once tasks share processors,
    so the true achievable energy may be somewhat higher in practice --
    this budget is a sound floor, not an exact prediction.
    """
    full_energy = 0.0
    mandatory_energy = 0.0

    for i, (e, p, m, k) in enumerate(tasks):
        n = eta[i]
        f_task = min_feasible_freq(e, p, freq_levels)
        job_cost = h_bar * (f_task ** 2) * e
        full_energy += n * job_cost
        mandatory_count = sum(
            1 for j in range(1, n + 1)
            if (j * m) // k != ((j - 1) * m) // k
        )
        mandatory_energy += mandatory_count * job_cost

    budget = mandatory_energy + slack_fraction * (full_energy - mandatory_energy)
    return round(budget, 2), mandatory_energy, full_energy


# =========================
# MAIN GENERATOR
# =========================

def generate_testcase(n_tsk, n_prc, n_frq, total_utilization,
                       p_min=4, ratio=None, k_range=(2, 4), tolerance="mixed",
                       freq_style="uniform", f_min=0.4, slack_fraction=0.4,
                       soft_warn_vars=20000, soft_warn_constrs=20000,
                       seed=None):
    """
    Generates one complete testcase.

    The ONLY hard structural requirement enforced is strict harmonic
    periods (generate_harmonic_periods), matching the paper's system
    model. There is no solver-license size cap -- soft_warn_vars/
    soft_warn_constrs just print a heads-up if the model is getting
    large enough that solve time may become noticeable; generation is
    never blocked by them.
    """
    if seed is not None:
        random.seed(seed)

    utils = uunifast_discard(n_tsk, total_utilization, u_max=1.0)
    periods, r = generate_harmonic_periods(n_tsk, p_min=p_min, ratio=ratio)
    mk_list = generate_mk(n_tsk, k_range=k_range, tolerance=tolerance)

    tasks = []
    for u_i, p_i, (m_i, k_i) in zip(utils, periods, mk_list):
        e_i = max(1, min(p_i, round(u_i * p_i)))
        tasks.append((e_i, p_i, m_i, k_i))

    h_eff = lcm_list([p * k for (e, p, m, k) in tasks])
    size = estimate_model_size(tasks, n_prc, n_frq, h_eff)

    if size["n_vars"] > soft_warn_vars or size["n_constrs"] > soft_warn_constrs:
        print(f"[testcase_generator] NOTE: this testcase is fairly large "
              f"({size['n_vars']} vars, {size['n_constrs']} constraints, "
              f"h_eff={h_eff}, {size['total_jobs']} jobs). Solve time may "
              f"be noticeable -- not blocked, just a heads-up.")

    freq_levels = generate_frequency_levels(n_frq, f_min=f_min, style=freq_style)
    energy_budget, mandatory_e, full_e = compute_energy_budget(
        tasks, size["eta"], freq_levels, slack_fraction=slack_fraction
    )

    info = {
        "ratio": r,
        "h_eff": h_eff,
        "eta": size["eta"],
        "n_vars": size["n_vars"],
        "n_constrs": size["n_constrs"],
        "breakdown": size["breakdown"],
        "mandatory_energy": mandatory_e,
        "full_energy": full_e,
    }

    return n_prc, freq_levels, energy_budget, tasks, info


def write_testcase(filename, n_prc, freq_levels, energy_budget, tasks):
    """
    filename: just the file's name (e.g. "t1.txt") -- ALWAYS written into
    TESTCASES_DIR, imported directly from preprocessing.py rather than
    re-derived here, so the generator and the solver can never disagree
    about where a testcase lives.

    Returns the resolved absolute path actually written, for logging.
    """
    TESTCASES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = TESTCASES_DIR / filename

    with open(filepath, "w") as f:
        f.write(f"{n_prc}\n")
        f.write(" ".join(str(x) for x in freq_levels) + "\n")
        f.write(f"{energy_budget}\n")
        for (e, p, m, k) in tasks:
            f.write(f"{e} {p} {m} {k}\n")

    return filepath


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    # Keep this filename in sync with model.py's TESTCASE_FILENAME --
    # both now point at the exact same resolved TESTCASES_DIR, so as
    # long as the string matches, there is no way for the two scripts
    # to end up looking at different files.
    TESTCASE_FILENAME = "t3.txt"

    n_prc, freq_levels, energy_budget, tasks, info = generate_testcase(
        n_tsk=5,
        n_prc=2,
        n_frq=4,
        total_utilization=1.2,
        p_min=4,
        ratio=2,
        k_range=(2, 4),
        tolerance="mixed",
        freq_style="uniform",
        f_min=0.4,
        slack_fraction=0.4,
        seed=random.seed(),
    )

    print("Generated tasks (e, p, m, k):")
    for t in tasks:
        print("  ", t)
    print("Frequency levels:", freq_levels)
    print("Energy budget:", energy_budget)
    print("h_eff:", info["h_eff"], "| eta per task:", info["eta"])
    print("Model size:", info["n_vars"], "vars,", info["n_constrs"],
          "constraints", info["breakdown"])
    print("Mandatory-only energy (per-task min feasible freq):",
          round(info["mandatory_energy"], 2))
    print("Full-acceptance energy (per-task min feasible freq):",
          round(info["full_energy"], 2))

    written_path = write_testcase(
        TESTCASE_FILENAME, n_prc, freq_levels, energy_budget, tasks
    )
    print(f"\nWritten to: {written_path}")