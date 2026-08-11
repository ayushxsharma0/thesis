"""
generate_test.py -- config-driven batch testcase generator for the EPFS
problem.

Usage:
    python generate_test.py [path/to/config.ini]
    (defaults to "config.ini" in the current directory if omitted)

Reads config.ini (see config format notes below), then for each of
num_testcases:
  1. Randomly samples every per-testcase parameter according to the
     config's notation ([a,b,c] = random pick, a-b = random range sample,
     x = fixed).
  2. Builds the task set (UUniFast-Discard utilization split, strict
     harmonic periods, tolerance-banded (m,k)) and an energy budget
     (mandatory-baseline + random alpha * slack), reusing the exact same,
     already-validated logic from testcase_generator.py.
  3. Retries with a fresh seed (up to max_retries_per_instance) if the
     draw is degenerate -- either infeasible-by-construction (a task's
     utilization exceeds every available frequency level) or simply too
     large (total job count over max_total_jobs).
  4. Writes successful testcases into testcases/<output_folder>/, and
     logs every attempt (success or final failure) to manifest.csv in
     that same folder, so every generated file is traceable back to
     exactly the parameters that produced it.

Config notation (see config.ini):
    field = [a, b, c]   -> per testcase, randomly pick ONE value from this set
    field = a-b         -> per testcase, randomly sample a value in this range
    field = x           -> fixed, same for every testcase in the batch
"""

import configparser
import csv
import random
import sys
from pathlib import Path

TESTCASES_DIR = Path(__file__).resolve().parent.parent / "testcases"
from testcase_generator import (
    uunifast_discard,
    generate_harmonic_periods,
    lcm_list,
    estimate_model_size,
    compute_energy_budget,
    min_feasible_freq,
)


# =========================
# CONFIG VALUE PARSING
# =========================
# configparser reads every value as a raw string. These helpers interpret
# our small custom notation on top of that.

def _parse_spec(raw, cast):
    """
    Parses a single config value into one of:
      ('fixed', value)
      ('list',  [values])
      ('range', (low, high))
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
    Special parser for utilization_bands: a list of RANGE strings, e.g.
    "[0.2-0.4, 0.4-0.6, 0.6-0.9]" -> [(0.2,0.4), (0.4,0.6), (0.6,0.9)]
    """
    s = raw.strip()[1:-1]
    bands = []
    for token in s.split(","):
        lo, hi = token.strip().split("-")
        bands.append((float(lo), float(hi)))
    return bands


def sample(spec, rng, as_int=False):
    """
    Draws one value from a parsed spec using the given RNG.
    """
    kind, val = spec
    if kind == "fixed":
        return val
    if kind == "list":
        return rng.choice(val)
    if kind == "range":
        lo, hi = val
        return rng.randint(lo, hi) if as_int else rng.uniform(lo, hi)
    raise ValueError(f"Unrecognized spec kind: {kind}")


# =========================
# LOAD CONFIG
# =========================

def load_config(path):
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    g = cp["general"]
    p = cp["processors"]
    t = cp["tasks"]

    cfg = {
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
    return cfg


# =========================
# PER-TESTCASE GENERATION
# =========================

def generate_frequency_levels(n_frq, f_min, f_max):
    """
    n_frq evenly spaced levels between f_min and f_max (inclusive).
    (Local helper, since testcase_generator.generate_frequency_levels
    hardcodes the top level at 1.0 -- here f_max is config-driven.)
    """
    if n_frq == 1:
        return [f_max]
    step = (f_max - f_min) / (n_frq - 1)
    return [round(f_min + i * step, 3) for i in range(n_frq)]


def sample_mk_for_task(rng, mk_type, ratio_lo, ratio_hi, k_min, k_max):
    """
    Samples (m, k) for one task. k is drawn uniformly from [k_min, k_max].
    m is derived from an m/k ratio drawn from a sub-band of
    [ratio_lo, ratio_hi] depending on mk_type:
      dense  -> upper third  (few misses tolerated)
      sparse -> lower third  (many misses tolerated)
      mixed  -> full range
    """
    k = rng.randint(k_min, k_max)
    third = (ratio_hi - ratio_lo) / 3.0

    if mk_type == "dense":
        lo, hi = ratio_hi - third, ratio_hi
    elif mk_type == "sparse":
        lo, hi = ratio_lo, ratio_lo + third
    else:  # mixed
        lo, hi = ratio_lo, ratio_hi

    ratio = rng.uniform(lo, hi)
    m = max(1, min(k, round(ratio * k)))
    return m, k


def attempt_generate_one(cfg, seed):
    """
    One random draw of a complete testcase. Returns a dict with the
    testcase data + metadata on success, or None if this draw is
    degenerate (infeasible-by-construction or too large) and should be
    retried with a different seed.
    """
    random.seed(seed)  # testcase_generator's reused functions use the
                        # global random module, so seed it globally for
                        # full reproducibility of this attempt.
    rng = random  # alias for readability below

    n_prc = sample(cfg["num_processors"], rng, as_int=True)
    n_frq = sample(cfg["freq_steps"], rng, as_int=True)
    freq_levels = generate_frequency_levels(n_frq, cfg["freq_min"], cfg["freq_max"])

    num_tasks = sample(cfg["num_tasks"], rng, as_int=True)

    band = rng.choice(cfg["utilization_bands"])
    total_utilization = rng.uniform(*band)

    period_ratio = sample(cfg["period_ratio"], rng, as_int=True)

    mk_type = sample(cfg["mk_type"], rng)
    ratio_lo, ratio_hi = cfg["mk_ratio_range"][1]  # fixed bounds to subdivide, not a value to sample

    # utilization split + harmonic periods
    utils = uunifast_discard(num_tasks, total_utilization, u_max=1.0)
    periods, actual_ratio = generate_harmonic_periods(
        num_tasks, p_min=cfg["p_min"], ratio=period_ratio
    )

    tasks = []
    for u_i, p_i in zip(utils, periods):
        e_i = max(1, min(p_i, round(u_i * p_i)))
        m_i, k_i = sample_mk_for_task(
            rng, mk_type, ratio_lo, ratio_hi, cfg["k_min"], cfg["k_max"]
        )
        tasks.append((e_i, p_i, m_i, k_i))

    h_eff = lcm_list([p * k for (e, p, m, k) in tasks])

    # Cheap pre-check BEFORE the expensive model-size estimate below --
    # estimate_model_size() builds full arrival/deadline sets, an
    # O(total_jobs) loop. A large period_ratio + several tasks can push
    # h_eff (and therefore total_jobs) into the millions on a single
    # unlucky draw; rejecting on this cheap sum-of-etas check first
    # avoids ever running that expensive loop on a doomed draw.
    quick_eta = [h_eff // p for (e, p, m, k) in tasks]
    if sum(quick_eta) > cfg["max_total_jobs"]:
        return None

    size = estimate_model_size(tasks, n_prc, n_frq, h_eff)

    energy_alpha = sample(cfg["energy_alpha"], rng)

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


# =========================
# WRITE ONE TESTCASE FILE
# =========================

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
                "filename": "",
                "status": "failed",
                "attempts_used": attempts_used,
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

    # write manifest.csv
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