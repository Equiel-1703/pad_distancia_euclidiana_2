import subprocess
import re
import csv
import numpy as np
from scipy import stats

executions = 30

t_vals = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# t_vals = [32] # only for testing

N_VECTORS = 3584  # LEN_X, fixed number of vectors in the x set
BOOTSTRAP_RESAMPLES = 9999
rng = np.random.default_rng(42)  # fixed seed for reproducibility across runs

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


def collect_samples(T, fast, n, raw_rows):
    samples = []
    label = "optimised" if fast else "normal"
    for i in range(n):
        t = run_once(T, fast=fast)
        samples.append(t)
        print(f"  [{label}] run {i + 1}/{n}: {t}ms")
        raw_rows.append({"T": T, "variant": label, "run": i + 1, "time_ms": t})
    return np.array(samples)


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std


def rank_biserial(a, b, u_stat):
    n1, n2 = len(a), len(b)
    return 1 - (2 * u_stat) / (n1 * n2)


def bootstrap_ci_95(samples):
    """
    95% CI for the mean via BCa bootstrap (bias-corrected and accelerated).
    Makes no assumption of normality, which suits skewed timing data
    with a small (n=30) sample.
    """
    res = stats.bootstrap(
        (samples,),
        statistic=np.mean,
        n_resamples=BOOTSTRAP_RESAMPLES,
        confidence_level=0.95,
        method="BCa",
        rng=rng,
    )
    return res.confidence_interval.low, res.confidence_interval.high


def throughput_metrics(samples, T):
    """
    Pv (vectors/sec) and Pe (elements/sec), based on the average execution time.
    t must be converted from ms to seconds before dividing.
    """
    t_seconds = np.mean(samples) / 1000.0
    pv = N_VECTORS / t_seconds
    pe = (N_VECTORS * T) / t_seconds
    return pv, pe


def variant_stats(samples, T):
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    median = np.median(samples)
    ci_low, ci_high = bootstrap_ci_95(samples)
    pv, pe = throughput_metrics(samples, T)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pv": pv,
        "pe": pe,
    }


results = []
raw_rows = []

for T in t_vals:
    print(f"\n=== T = {T} ===")

    print("Running normal version...")
    normal_samples = collect_samples(T, fast=False, n=executions, raw_rows=raw_rows)

    print("Running optimised version...")
    fast_samples = collect_samples(T, fast=True, n=executions, raw_rows=raw_rows)

    normal_stats = variant_stats(normal_samples, T)
    fast_stats = variant_stats(fast_samples, T)

    mw_result = stats.mannwhitneyu(normal_samples, fast_samples, alternative="two-sided")
    p_value = mw_result.pvalue

    d = cohens_d(normal_samples, fast_samples)
    r = rank_biserial(normal_samples, fast_samples, mw_result.statistic)

    significant = p_value < 0.05
    faster = fast_stats["mean"] < normal_stats["mean"]

    summary = {
        "T": T,
        "mean_normal": normal_stats["mean"],
        "std_normal": normal_stats["std"],
        "median_normal": normal_stats["median"],
        "ci95_low_normal": normal_stats["ci_low"],
        "ci95_high_normal": normal_stats["ci_high"],
        "pv_normal": normal_stats["pv"],
        "pe_normal": normal_stats["pe"],
        "mean_fast": fast_stats["mean"],
        "std_fast": fast_stats["std"],
        "median_fast": fast_stats["median"],
        "ci95_low_fast": fast_stats["ci_low"],
        "ci95_high_fast": fast_stats["ci_high"],
        "pv_fast": fast_stats["pv"],
        "pe_fast": fast_stats["pe"],
        "p_value": p_value,
        "significant": significant,
        "faster": faster,
        "cohens_d": d,
        "rank_biserial_r": r,
    }
    results.append(summary)

    print(f"\n--- Results for T = {T} ---")
    print(f"Normal:    mean={normal_stats['mean']:.5f}ms  std={normal_stats['std']:.5f}ms  "
          f"median={normal_stats['median']:.5f}ms  "
          f"95% CI (bootstrap)=[{normal_stats['ci_low']:.5f}, {normal_stats['ci_high']:.5f}]ms")
    print(f"           Pv={normal_stats['pv']:.2f} vectors/s  Pe={normal_stats['pe']:.2f} elements/s")
    print(f"Optimised: mean={fast_stats['mean']:.5f}ms  std={fast_stats['std']:.5f}ms  "
          f"median={fast_stats['median']:.5f}ms  "
          f"95% CI (bootstrap)=[{fast_stats['ci_low']:.5f}, {fast_stats['ci_high']:.5f}]ms")
    print(f"           Pv={fast_stats['pv']:.2f} vectors/s  Pe={fast_stats['pe']:.2f} elements/s")
    print(f"Mann-Whitney U: p={p_value:.5f}")
    print(f"Cohen's d: {d:.4f}")
    print(f"Rank-biserial r: {r:.4f}")

    if significant:
        direction = "optimised was significantly faster" if faster else "optimised was significantly slower"
        print(f"=> {direction} (p < 0.05)")
    else:
        print("=> No statistically significant difference")

# Final summary table (console)
print("\n\n=== SUMMARY ===")
print(f"{'T':>6} | {'mean_normal':>12} | {'mean_fast':>12} | {'p_value':>8} | {'cohens_d':>8} | {'rank_r':>8} | conclusion")
for r in results:
    conclusion = "no diff"
    if r["significant"]:
        conclusion = "faster" if r["faster"] else "slower"
    print(f"{r['T']:>6} | {r['mean_normal']:>12.5f} | {r['mean_fast']:>12.5f} | {r['p_value']:>8.5f} | {r['cohens_d']:>8.4f} | {r['rank_biserial_r']:>8.4f} | {conclusion}")

# --- CSV output ---

# Raw per-run timings
with open("raw_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["T", "variant", "run", "time_ms"])
    writer.writeheader()
    writer.writerows(raw_rows)

# Per-T summary stats
with open("summary_results.csv", "w", newline="") as f:
    fieldnames = [
        "T",
        "mean_normal", "std_normal", "median_normal", "ci95_low_normal", "ci95_high_normal",
        "pv_normal", "pe_normal",
        "mean_fast", "std_fast", "median_fast", "ci95_low_fast", "ci95_high_fast",
        "pv_fast", "pe_fast",
        "p_value", "significant", "faster", "cohens_d", "rank_biserial_r",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\nSaved raw_results.csv and summary_results.csv")
