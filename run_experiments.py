import subprocess
import re
import numpy as np
from scipy import stats

executions = 30
# t_vals = [32, 64, 128, 256, 512, 1024, 2048, 4096]
t_vals = [32]

# Ensuring we have compiled the code first
subprocess.run(["cmake", "-S", ".", "-B", "build"])
subprocess.run(["cmake", "--build", "build"])


def run_once(T, fast=False):
    args = ["./build/distancia_euclidiana", str(T)]
    if fast:
        args.append("-f")

    result = subprocess.run(args, capture_output=True, text=True)
    output = result.stdout.strip()

    match = re.search(r"\|\s*([\d.]+)ms", output)
    if not match:
        raise ValueError(f"Could not parse output: '{output}'")

    return float(match.group(1))


def collect_samples(T, fast, n):
    samples = []
    for i in range(n):
        t = run_once(T, fast=fast)
        samples.append(t)
        label = "optimised" if fast else "normal"
        print(f"  [{label}] run {i + 1}/{n}: {t}ms")
    return np.array(samples)


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std


def rank_biserial(a, b, u_stat):
    # r = 1 - 2U / (n1 * n2)
    # u_stat here is U for `a` vs `b` (from mannwhitneyu(a, b, ...))
    n1, n2 = len(a), len(b)
    return 1 - (2 * u_stat) / (n1 * n2)


results = []

for T in t_vals:
    print(f"\n=== T = {T} ===")

    print("Running normal version...")
    normal_samples = collect_samples(T, fast=False, n=executions)

    print("Running optimised version...")
    fast_samples = collect_samples(T, fast=True, n=executions)

    mean_normal, std_normal = np.mean(normal_samples), np.std(normal_samples, ddof=1)
    mean_fast, std_fast = np.mean(fast_samples), np.std(fast_samples, ddof=1)

    mw_result = stats.mannwhitneyu(normal_samples, fast_samples, alternative="two-sided")
    p_value = mw_result.pvalue

    d = cohens_d(normal_samples, fast_samples)
    r = rank_biserial(normal_samples, fast_samples, mw_result.statistic)

    significant = p_value < 0.05
    faster = mean_fast < mean_normal

    summary = {
        "T": T,
        "mean_normal": mean_normal,
        "std_normal": std_normal,
        "mean_fast": mean_fast,
        "std_fast": std_fast,
        "p_value": p_value,
        "significant": significant,
        "faster": faster,
        "cohens_d": d,
        "rank_biserial_r": r,
    }
    results.append(summary)

    print(f"\n--- Results for T = {T} ---")
    print(f"Normal:    mean={mean_normal:.5f}ms  std={std_normal:.5f}ms")
    print(f"Optimised: mean={mean_fast:.5f}ms  std={std_fast:.5f}ms")
    print(f"Mann-Whitney U: p={p_value:.5f}")
    print(f"Cohen's d: {d:.4f}")
    print(f"Rank-biserial r: {r:.4f}")

    if significant:
        direction = "optimised was significantly faster" if faster else "optimised was significantly slower"
        print(f"=> {direction} (p < 0.05)")
    else:
        print("=> No statistically significant difference")

# Final summary table
print("\n\n=== SUMMARY ===")
print(f"{'T':>6} | {'mean_normal':>12} | {'mean_fast':>12} | {'p_value':>8} | {'cohens_d':>8} | {'rank_r':>8} | conclusion")
for r in results:
    conclusion = "no diff"
    if r["significant"]:
        conclusion = "faster" if r["faster"] else "slower"
    print(f"{r['T']:>6} | {r['mean_normal']:>12.5f} | {r['mean_fast']:>12.5f} | {r['p_value']:>8.5f} | {r['cohens_d']:>8.4f} | {r['rank_biserial_r']:>8.4f} | {conclusion}")