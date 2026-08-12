"""
generate_testcase.py -- master file for config-driven batch testcase
generation.

Usage:
    python generate_testcase.py [path/to/config.ini]
    (defaults to "config.ini" in this folder if omitted)

Orchestrates the four worker modules (periods.py, utilization.py,
mk_pattern.py, processors.py), each of which does one piece of pure
generation given already-resolved parameters. This file owns everything
that ISN'T a single worker's concern:
  - config.ini parsing and the [a,b,c]/a-b/x notation interpretation
  - per-testcase random sampling of which concrete value each worker
    module should use
  - assembling the four pieces into a complete task set
  - energy budget calculation (needs tasks AND freq_levels together, so
    it doesn't belong to any one worker module)
  - model-size estimation and the safety check against max_total_jobs
  - the retry/skip/abort loop and manifest.csv logging
  - writing the final testcase files in preprocessing.py's expected format

Folder layout assumption: this file lives in testcase_generator/, a
SIBLING of ILP/ under the project root, so that
Path(__file__).parent.parent resolves to the project root and
/ "testcases" lands at project_root/testcases -- a sibling of both
testcase_generator/ and ILP/. preprocessing.py (in ILP/) resolves the
exact same folder via its own, independent copy of this same relative
path convention -- the two are not imported from one another (avoiding
the cross-folder import problem that broke the previous layout), but as
long as both files stay at this same folder depth, they always agree.

Config notation (see config.ini):
    field = [a, b, c]   -> per testcase, randomly pick ONE value from this set
    field = a-b         -> per testcase, randomly sample a value in this range
    field = x           -> fixed, same for every testcase in the batch
"""

import configparser
import csv
import math
import random
import sys
from pathlib import Path

from periods import generate_harmonic_periods
from utilization import uunifast_discard
from mk_pattern import sample_mk_for_task
from processors import generate_frequency_levels


# =========================
# SHARED PATH
# =========================

TESTCASES_DIR = Path(__file__).resolve().parent.parent / "testcases"


# =========================
# MATH HELPERS (master-only concerns: model sizing, energy pricing)
# =========================

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def lcm_list(values):
    result = values[0]
    for v in values[1:]:
        result = lcm(result, v)
    return result


def min_feasible_freq(e_i, p_i, freq_levels):
    """
    Cheapest frequency at which a job can meet ITS OWN deadline in
    isolation (f >= e_i/p_i). A high-utilization task may be barred from
    every "cheap" frequency level -- pricing it as if it could use them
    anyway produces an energy budget that's infeasible by construction.
    """
    u_i = e_i / p_i
    feasible = [f for f in freq_levels if f >= u_i - 1e-9]
    if not feasible:
        raise ValueError(
            f"Task with e={e_i}, p={p_i} has utilization {u_i:.3f}, which "
            f"exceeds every available frequency level {freq_levels}."
        )
    return min(feasible)


def compute_energy_budget(tasks, eta, freq_levels, slack_fraction, h_bar=1.0):
    """
    energy_budget = mandatory_energy + slack_fraction * (full_energy - mandatory_energy),
    each priced at every task's OWN minimum feasible frequency (not a
    shared global minimum -- see min_feasible_freq).
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


def estimate_model_size(tasks, n_prc, n_frq, h_eff):
    """
    Mirrors model.py's exact C1-C4 constraint construction, so the
    reported size matches what Gurobi would actually build.
    """
    eta = [h_eff // p for (e, p, m, k) in tasks]
    total_jobs = sum(eta)

    n_vars = total_jobs * n_prc * n_frq + total_jobs

    c1 = 2 * total_jobs

    c3 = 0
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

    return {
        "eta": eta,
        "total_jobs": total_jobs,
        "n_vars": n_vars,
        "n_constrs": c1 + c2 + c3 + c4,
        "breakdown": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
    }


# =========================
# CONFIG VALUE PARSING
# =========================

def _parse_spec(raw, cast):
    """
    Parses a single config value into one of:
      ('fixed', value)  ('list', [values])  ('range', (low, high))
    """
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        items = [x.strip() for x in s[1:-1].split(",")]
        return ("list", [cast(x) for x in items])
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2:
            return ("range", (cast(parts[0]), cast(parts[1])))
    return ("fixed", cast(s))


def _parse_band_list(raw):
    """
    utilization_bands: a list of RANGE strings, e.g.
    "[0.2-0.4, 0.4-0.6, 0.6-0.9]" -> [(0.2,0.4), (0.4,0.6), (0.6,0.9)]
    """
    s = raw.strip()[1:-1]
    bands = []
    for token in s.split(","):
        lo, hi = token.strip().split("-")
        bands.append((float(lo), float(hi)))
    return bands


def sample(spec, as_int=False):
    """Draws one value from a parsed spec using the global random module."""
    kind, val = spec
    if kind == "fixed":
        return val
    if kind == "list":
        return random.choice(val)
    if kind == "range":
        lo, hi = val
        return random.randint(lo, hi) if as_int else random.uniform(lo, hi)
    raise ValueError(f"Unrecognized spec kind: {kind}")


def load_config(path):
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    g = cp["general"]
    p = cp["processors"]
    t = cp["tasks"]

    return {
        "num_testcases": int(g["num_testcases"]),
        "base_seed": int(g["base_seed"]),
        "max_retries_per_instance": int(g["max_retries_per_instance"]),
        "on_failure": g["on_failure"].strip(),
        "max_total_jobs": int(g["max_total_jobs"]),
        "output_folder": g["output_folder"].strip(),
        "manifest_file": g["manifest_file"].strip(),

        "num_processors": _parse_spec(p["num_processors"], int),
        "freq_min": float(p["freq_min"]),
        "freq_max": float(p["freq_max"]),
        "freq_steps": _parse_spec(p["freq_steps"], int),
        "energy_alpha": _parse_spec(p["energy_alpha"], float),

        "num_tasks": _parse_spec(t["num_tasks"], int),
        "utilization_bands": _parse_band_list(t["utilization_bands"]),
        "period_ratio": _parse_spec(t["period_ratio"], int),
        "p_min": int(t["p_min"]),
        "mk_type": _parse_spec(t["mk_type"], str),
        "mk_ratio_range": _parse_spec(t["mk_ratio_range"], float),
        "k_min": int(t["k_min"]),
        "k_max": int(t["k_max"]),
    }


# =========================
# PER-TESTCASE GENERATION
# =========================

def attempt_generate_one(cfg, seed):
    """
    One random draw of a complete testcase, calling out to the 4 worker
    modules for their respective pieces. Returns a result dict on
    success, or None if this draw is degenerate and should be retried.
    """
    random.seed(seed)

    n_prc = sample(cfg["num_processors"], as_int=True)
    n_frq = sample(cfg["freq_steps"], as_int=True)
    freq_levels = generate_frequency_levels(n_frq, cfg["freq_min"], cfg["freq_max"])

    num_tasks = sample(cfg["num_tasks"], as_int=True)

    band = random.choice(cfg["utilization_bands"])
    total_utilization = random.uniform(*band)

    period_ratio = sample(cfg["period_ratio"], as_int=True)

    mk_type = sample(cfg["mk_type"])
    ratio_lo, ratio_hi = cfg["mk_ratio_range"][1]  # fixed sub-band bounds, not sampled

    utils = uunifast_discard(num_tasks, total_utilization, u_max=1.0)
    periods = generate_harmonic_periods(num_tasks, p_min=cfg["p_min"], ratio=period_ratio)

    tasks = []
    for u_i, p_i in zip(utils, periods):
        e_i = max(1, min(p_i, round(u_i * p_i)))
        m_i, k_i = sample_mk_for_task(mk_type, ratio_lo, ratio_hi, cfg["k_min"], cfg["k_max"])
        tasks.append((e_i, p_i, m_i, k_i))

    h_eff = lcm_list([p * k for (e, p, m, k) in tasks])

    # Cheap pre-check BEFORE the expensive model-size estimate below --
    # estimate_model_size() builds full arrival/deadline sets, an
    # O(total_jobs) loop, so reject an oversized draw before running it.
    quick_eta = [h_eff // p for (e, p, m, k) in tasks]
    if sum(quick_eta) > cfg["max_total_jobs"]:
        return None

    size = estimate_model_size(tasks, n_prc, n_frq, h_eff)

    energy_alpha = sample(cfg["energy_alpha"])

    try:
        energy_budget, mandatory_e, full_e = compute_energy_budget(
            tasks, size["eta"], freq_levels, slack_fraction=energy_alpha
        )
    except ValueError:
        return None  # a task's utilization exceeds every frequency level -- retry

    return {
        "n_prc": n_prc,
        "freq_levels": freq_levels,
        "energy_budget": energy_budget,
        "tasks": tasks,
        "seed": seed,
        "num_tasks": num_tasks,
        "total_utilization": round(total_utilization, 4),
        "utilization_band": f"{band[0]}-{band[1]}",
        "period_ratio": period_ratio,
        "mk_type": mk_type,
        "energy_alpha": round(energy_alpha, 4),
        "mandatory_energy": round(mandatory_e, 4),
        "full_energy": round(full_e, 4),
        "h_eff": h_eff,
        "total_jobs": size["total_jobs"],
        "n_vars": size["n_vars"],
        "n_constrs": size["n_constrs"],
    }


def write_testcase_file(filepath, result):
    with open(filepath, "w") as f:
        f.write(f"{result['n_prc']}\n")
        f.write(" ".join(str(x) for x in result["freq_levels"]) + "\n")
        f.write(f"{result['energy_budget']}\n")
        for (e, p, m, k) in result["tasks"]:
            f.write(f"{e} {p} {m} {k}\n")


# =========================
# MAIN
# =========================

def main(config_path):
    cfg = load_config(config_path)

    out_dir = TESTCASES_DIR / cfg["output_folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / cfg["manifest_file"]

    manifest_rows = []
    seed_counter = cfg["base_seed"]
    n_generated = 0
    n_skipped = 0

    for idx in range(1, cfg["num_testcases"] + 1):
        result = None
        attempts_used = 0

        for attempt in range(cfg["max_retries_per_instance"] + 1):
            attempts_used += 1
            result = attempt_generate_one(cfg, seed_counter)
            seed_counter += 1
            if result is not None:
                break

        if result is None:
            n_skipped += 1
            print(f"[{idx}/{cfg['num_testcases']}] FAILED after "
                  f"{attempts_used} attempts (skipped).")
            manifest_rows.append({
                "filename": "", "status": "failed", "attempts_used": attempts_used,
            })
            if cfg["on_failure"] == "abort":
                print("on_failure=abort -- stopping batch.")
                break
            continue

        filename = f"t{idx:04d}.txt"
        filepath = out_dir / filename
        write_testcase_file(filepath, result)
        n_generated += 1

        print(f"[{idx}/{cfg['num_testcases']}] OK -> {filename} "
              f"(n_prc={result['n_prc']}, n_tsk={result['num_tasks']}, "
              f"util={result['total_utilization']}, h_eff={result['h_eff']}, "
              f"jobs={result['total_jobs']}, attempts={attempts_used})")

        row = {"filename": filename, "status": "ok", "attempts_used": attempts_used}
        row.update({k: v for k, v in result.items() if k not in ("freq_levels", "tasks")})
        row["freq_levels"] = ";".join(str(x) for x in result["freq_levels"])
        row["tasks"] = ";".join(f"({e},{p},{m},{k})" for (e, p, m, k) in result["tasks"])
        manifest_rows.append(row)

    if manifest_rows:
        all_keys = []
        for row in manifest_rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\nDone. {n_generated} generated, {n_skipped} skipped, "
          f"out of {cfg['num_testcases']} requested.")
    print(f"Output folder: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    main(config_arg)