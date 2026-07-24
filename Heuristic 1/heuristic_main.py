from preprocessing import preprocess
from pattern import generate_pattern
from sps import run_sps
from fairness import run_fairness


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
    freq_indices
) = preprocess("..\\testcases\\t3.txt")

time_pairs = [
    (t1, t2) for t1 in arrival_times for t2 in deadline_times if t2 > t1
]


# =========================
# STAGE 1: PATTERN (DPS)
# =========================
pattern = generate_pattern(tasks, jobs)


# =========================
# STAGE 2: SPS
# =========================
state, demoted = run_sps(
    tasks, jobs, pattern, time_pairs, energy_budget, freq_levels, n_prc
)

if demoted:
    print(f"[SPS] {len(demoted)} job(s) demoted from accept -> skip "
          f"due to DBF/energy conflicts: {demoted}")


# =========================
# STAGE 3: FAIRNESS
# =========================
state = run_fairness(state, tasks, jobs, eta)


# =========================
# OUTPUT
# =========================
print("\n=== Heuristic Solution ===")

for i in range(len(tasks)):
    for j in jobs[i]:
        placement = state.assignment[(i, j)]
        if placement is not None:
            x, f = placement
            print(
                f"Task {i}, Job {j}"
                f" -> Processor {x}"
                f" @ Frequency {freq_levels[f]}"
            )
        else:
            print(f"Task {i}, Job {j} -> SKIPPED")

print()
gammas = []
for i in range(len(tasks)):
    accepted = sum(1 for j in jobs[i] if state.assignment[(i, j)] is not None)
    gamma_i = (eta[i] - accepted) / eta[i]
    gammas.append(gamma_i)
    print(f"Task {i}: eta={eta[i]}, accepted={accepted}, gamma={gamma_i:.4f}")

phi = max(gammas)
print("\nPhi (heuristic):", phi)
print("Total Energy (heuristic):", state.total_energy)

heuristic_phi = phi
heuristic_patterns = {
    i: [1 if state.assignment[(i, j)] is not None else 0 for j in jobs[i]]
    for i in range(len(tasks))
}
