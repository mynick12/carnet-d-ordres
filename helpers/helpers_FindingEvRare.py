import math
import contextlib
import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Acer\Documents\GitHub\carnet-d-ordres")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from helpers.helpers_Splitting import *
from helpers.helpers_QR2LIM import *

def naive_reactive_sanity_check(
    *,
    q0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    b_second,
    gamma_cross=1.0,
    t_max=10_000.0,
    n_samples=3_000,
    a_second=1.0,
    h_values=None,
    seed=123,
):
    """
    Naive Monte Carlo sanity check for the reactive second-limit model.

    Estimates:
      P(N^{+1}_tau = 0)
      P(N^{+2}_tau <= h | N^{+1}_tau = 0)
      P(N^{+1}_tau = 0, N^{+2}_tau <= h)

    This is only a sanity check, not the main rare-event estimator.
    """
    if h_values is None:
        h_values = np.arange(0, q2_0 + 1)

    h_values = np.asarray(h_values, dtype=int)
    rng = np.random.default_rng(seed)

    out = sample_second_limit_conditionals_reactive(
        n_samples=n_samples,
        q1_0=q0,
        q2_0=q2_0,
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        alpha=alpha,
        beta=beta,
        lambda_second_add=lambda_second_add,
        lambda_second_remove=lambda_second_remove,
        a_second=a_second,
        b_second=b_second,
        gamma_cross=gamma_cross,
        rng=rng,
        t_max=t_max,
    )

    first_empty = out["first_empty"]
    hit_zero = out["hit_zero"]
    plus2_at_tau = out["plus2_at_tau"]

    plus1_empty_mask = hit_zero & (first_empty == "plus1")
    plus2_given_plus1_empty = plus2_at_tau[plus1_empty_mask]

    rows = []

    for h in h_values:
        if len(plus2_given_plus1_empty) == 0:
            p_cond = np.nan
            p_joint = 0.0
        else:
            p_cond = np.mean(plus2_given_plus1_empty <= h)
            p_joint = np.mean(plus1_empty_mask & (plus2_at_tau <= h))

        rows.append({
            "h": int(h),
            "n_samples": int(n_samples),
            "n_plus1_empty": int(np.sum(plus1_empty_mask)),
            "p_hit_plus1_zero_naive": float(np.mean(plus1_empty_mask)),
            "p_cond_second_low_naive": float(p_cond) if not np.isnan(p_cond) else np.nan,
            "p_joint_naive": float(p_joint),
            "a_second": float(a_second),
        })

    return pd.DataFrame(rows), plus2_given_plus1_empty, out

def simulate_second_values_from_final_particles(
    final_particles,
    q2_0,
    lambda_second_add,
    lambda_second_remove,
    a_second,
    b_second,
    n_second_rep=3,
    seed=123,
):
    """
    Given final first-limit splitting particles, simulate N^{+2}_tau several
    times per final particle.

    This avoids rerunning the whole first-limit splitting when only the
    second-limit lower tail is being studied.
    """
    rng = np.random.default_rng(seed)
    plus2_values = []

    for p in final_particles:
        for _ in range(n_second_rep):
            q2_tau = simulate_reactive_second_limit_until_time(
                q2_0=q2_0,
                horizon=p["t"],
                driver_removal_times=p["bid_removal_times"],
                mu_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                a=a_second,
                b=b_second,
                rng=rng,
            )
            plus2_values.append(q2_tau)

    return np.asarray(plus2_values, dtype=float)


def run_splitting_estimator_quiet(
    *,
    h,
    a_second,
    n_particles,
    levels,
    q0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    b_second,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    seed=123,
    quiet=True,
):
    """
    Thin wrapper around the original helpers_Splitting.splitting_estimator.

    We use this to avoid relying on a notebook-defined run_splitting_estimator.
    """
    rng = np.random.default_rng(seed)

    kwargs = dict(
        h=h,
        n_particles=n_particles,
        levels=levels,
        q0=q0,
        q2_0=q2_0,
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        alpha=alpha,
        beta=beta,
        lambda_second_add=lambda_second_add,
        lambda_second_remove=lambda_second_remove,
        a_second=a_second,
        b_second=b_second,
        gamma_cross=gamma_cross,
        lambda_floor=lambda_floor,
        t_max=t_max,
        rng=rng,
    )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return splitting_estimator(**kwargs)

    return splitting_estimator(**kwargs)


def run_main_sensitivity_h_grid(
    *,
    a_second_values,
    h_values,
    q0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    b_second,
    levels=None,
    n_runs=3,
    n_particles=5_000,
    n_second_rep=3,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    base_seed=20_000,
    quiet=True,
):
    """
    Main sensitivity analysis.

    For each independent splitting run:
      1. Generate final particles conditioned on N^{+1}_tau = 0.
      2. For each a_second, simulate N^{+2}_tau several times per final particle.
      3. For every h in the same h-grid, estimate:
           P(N^{+2}_tau <= h | N^{+1}_tau = 0)
           P(N^{+1}_tau = 0, N^{+2}_tau <= h)

    The same h-grid is used for every a_second, so the effect of a_second
    can be compared cleanly.
    """
    if levels is None:
        levels = [8, 6, 4, 2, 0]

    rows = []

    a_second_values = np.asarray(a_second_values, dtype=float)
    h_values = np.asarray(h_values, dtype=int)

    # Dummy values only needed because splitting_estimator requires h and a_second.
    # The first-limit final_particles do not depend on h or a_second.
    h_dummy = int(np.max(h_values))
    a_dummy = float(a_second_values[0])

    for r in range(n_runs):
        seed_split = base_seed + r

        res = run_splitting_estimator_quiet(
            h=h_dummy,
            a_second=a_dummy,
            n_particles=n_particles,
            levels=levels,
            q0=q0,
            q2_0=q2_0,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            beta=beta,
            lambda_second_add=lambda_second_add,
            lambda_second_remove=lambda_second_remove,
            b_second=b_second,
            gamma_cross=gamma_cross,
            lambda_floor=lambda_floor,
            t_max=t_max,
            seed=seed_split,
            quiet=quiet,
        )

        p_hit = float(res["p_hit_plus1_zero"])
        final_particles = res["final_particles"]

        for a_idx, a_val in enumerate(a_second_values):
            seed_second = base_seed + 100_000 + 10_000 * r + a_idx

            plus2_values = simulate_second_values_from_final_particles(
                final_particles=final_particles,
                q2_0=q2_0,
                lambda_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                a_second=float(a_val),
                b_second=b_second,
                n_second_rep=n_second_rep,
                seed=seed_second,
            )

            n_second_simulations = len(plus2_values)

            for h_val in h_values:
                indicators = plus2_values <= h_val
                p_cond = float(np.mean(indicators))
                p_joint = p_hit * p_cond

                rows.append(
                    {
                        "run": int(r),
                        "seed_split": int(seed_split),
                        "a_second": float(a_val),
                        "h": int(h_val),
                        "n_particles": int(n_particles),
                        "n_second_rep": int(n_second_rep),
                        "n_second_simulations": int(n_second_simulations),
                        "n_low_hits": int(np.sum(indicators)),
                        "p_hit_plus1_zero": p_hit,
                        "p_cond_second_low": p_cond,
                        "p_joint": p_joint,
                        "survival_fractions": list(res["survival_fractions"]),
                    }
                )

    return pd.DataFrame(rows)


def summarize_main_sensitivity(df_sensitivity_runs):
    """
    Aggregate the main sensitivity runs by (a_second, h).
    """
    rows = []
    z = 1.96

    for (a_val, h_val), grp in df_sensitivity_runs.groupby(["a_second", "h"]):
        row = {
            "a_second": float(a_val),
            "h": int(h_val),
            "n_runs": int(len(grp)),
            "n_particles": int(grp["n_particles"].iloc[0]),
            "n_second_rep": int(grp["n_second_rep"].iloc[0]),
            "total_second_simulations": int(grp["n_second_simulations"].sum()),
            "total_low_hits": int(grp["n_low_hits"].sum()),
            "p_hit_plus1_zero_mean": float(grp["p_hit_plus1_zero"].mean()),
        }

        for col in ["p_cond_second_low", "p_joint"]:
            values = grp[col].to_numpy(dtype=float)

            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            half_width = z * sd / math.sqrt(len(values)) if len(values) > 1 else np.nan

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = sd

            if len(values) > 1:
                row[f"{col}_ci95_low"] = max(0.0, mean - half_width)
                row[f"{col}_ci95_high"] = mean + half_width
            else:
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["a_second", "h"]).reset_index(drop=True)


def repeated_splitting_runs_selected_pairs(
    *,
    selected_pairs,
    q0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    b_second,
    levels=None,
    n_runs=10,
    n_particles=8_000,
    n_second_rep=5,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    base_seed=12_345,
    quiet=True,
):
    """
    Expensive uncertainty analysis for selected (a_second, h) pairs only.
    """
    if levels is None:
        levels = [8, 6, 4, 2, 0]

    rows = []

    for pair_id, (a_val, h_val) in enumerate(selected_pairs):
        for r in range(n_runs):
            seed_split = base_seed + 1_000 * pair_id + r
            seed_second = base_seed + 100_000 + 1_000 * pair_id + r

            res = run_splitting_estimator_quiet(
                h=int(h_val),
                a_second=float(a_val),
                n_particles=n_particles,
                levels=levels,
                q0=q0,
                q2_0=q2_0,
                mu_plus=mu_plus,
                mu_minus=mu_minus,
                alpha=alpha,
                beta=beta,
                lambda_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                b_second=b_second,
                gamma_cross=gamma_cross,
                lambda_floor=lambda_floor,
                t_max=t_max,
                seed=seed_split,
                quiet=quiet,
            )

            plus2_values = simulate_second_values_from_final_particles(
                final_particles=res["final_particles"],
                q2_0=q2_0,
                lambda_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                a_second=float(a_val),
                b_second=b_second,
                n_second_rep=n_second_rep,
                seed=seed_second,
            )

            indicators = plus2_values <= h_val

            p_hit = float(res["p_hit_plus1_zero"])
            p_cond = float(np.mean(indicators))
            p_joint = p_hit * p_cond

            rows.append(
                {
                    "a_second": float(a_val),
                    "h": int(h_val),
                    "run": int(r),
                    "seed_split": int(seed_split),
                    "seed_second": int(seed_second),
                    "n_particles": int(n_particles),
                    "n_second_rep": int(n_second_rep),
                    "n_second_simulations": int(len(plus2_values)),
                    "n_low_hits": int(np.sum(indicators)),
                    "p_hit_plus1_zero": p_hit,
                    "p_cond_second_low": p_cond,
                    "p_joint": p_joint,
                    "survival_fractions": list(res["survival_fractions"]),
                }
            )

    return pd.DataFrame(rows)


def summarize_repeated_runs_selected_pairs(df_runs):
    rows = []
    z = 1.96

    for (a_val, h_val), grp in df_runs.groupby(["a_second", "h"]):
        row = {
            "a_second": float(a_val),
            "h": int(h_val),
            "n_runs": int(len(grp)),
            "n_particles": int(grp["n_particles"].iloc[0]),
            "n_second_rep": int(grp["n_second_rep"].iloc[0]),
            "total_second_simulations": int(grp["n_second_simulations"].sum()),
            "total_low_hits": int(grp["n_low_hits"].sum()),
        }

        for col in ["p_cond_second_low", "p_joint"]:
            values = grp[col].to_numpy(dtype=float)

            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            half_width = z * sd / math.sqrt(len(values)) if len(values) > 1 else np.nan

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = sd

            if len(values) > 1:
                row[f"{col}_ci95_low"] = max(0.0, mean - half_width)
                row[f"{col}_ci95_high"] = mean + half_width
            else:
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan

            if row["total_low_hits"] == 0:
                row[f"{col}_comment"] = "zero low-tail hits; below current resolution"
            else:
                row[f"{col}_comment"] = ""

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["a_second", "h"]).reset_index(drop=True)

def repeated_splitting_runs_selected_pairs(
    *,
    selected_pairs,
    q0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    b_second,
    levels=None,
    n_runs=10,
    n_particles=8_000,
    n_second_rep=5,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    base_seed=12_345,
    quiet=True,
):
    """
    Expensive uncertainty analysis for selected (a_second, h) pairs only.
    """
    if levels is None:
        levels = [8, 6, 4, 2, 0]

    rows = []

    for pair_id, (a_val, h_val) in enumerate(selected_pairs):
        for r in range(n_runs):
            seed_split = base_seed + 1_000 * pair_id + r
            seed_second = base_seed + 100_000 + 1_000 * pair_id + r

            res = run_splitting_estimator_quiet(
                h=int(h_val),
                a_second=float(a_val),
                n_particles=n_particles,
                levels=levels,
                q0=q0,
                q2_0=q2_0,
                mu_plus=mu_plus,
                mu_minus=mu_minus,
                alpha=alpha,
                beta=beta,
                lambda_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                b_second=b_second,
                gamma_cross=gamma_cross,
                lambda_floor=lambda_floor,
                t_max=t_max,
                seed=seed_split,
                quiet=quiet,
            )

            plus2_values = simulate_second_values_from_final_particles(
                final_particles=res["final_particles"],
                q2_0=q2_0,
                lambda_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                a_second=float(a_val),
                b_second=b_second,
                n_second_rep=n_second_rep,
                seed=seed_second,
            )

            indicators = plus2_values <= h_val

            p_hit = float(res["p_hit_plus1_zero"])
            p_cond = float(np.mean(indicators))
            p_joint = p_hit * p_cond

            rows.append({
                "a_second": float(a_val),
                "h": int(h_val),
                "run": int(r),
                "seed_split": int(seed_split),
                "seed_second": int(seed_second),
                "n_particles": int(n_particles),
                "n_second_rep": int(n_second_rep),
                "n_second_simulations": int(len(plus2_values)),
                "n_low_hits": int(np.sum(indicators)),
                "p_hit_plus1_zero": p_hit,
                "p_cond_second_low": p_cond,
                "p_joint": p_joint,
                "survival_fractions": list(res["survival_fractions"]),
            })

    return pd.DataFrame(rows)


def summarize_repeated_runs_selected_pairs(df_runs):
    rows = []
    z = 1.96

    for (a_val, h_val), grp in df_runs.groupby(["a_second", "h"]):
        row = {
            "a_second": float(a_val),
            "h": int(h_val),
            "n_runs": int(len(grp)),
            "n_particles": int(grp["n_particles"].iloc[0]),
            "n_second_rep": int(grp["n_second_rep"].iloc[0]),
            "total_second_simulations": int(grp["n_second_simulations"].sum()),
            "total_low_hits": int(grp["n_low_hits"].sum()),
        }

        for col in ["p_cond_second_low", "p_joint"]:
            values = grp[col].to_numpy(dtype=float)

            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            half_width = z * sd / math.sqrt(len(values)) if len(values) > 1 else np.nan

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = sd

            if len(values) > 1:
                row[f"{col}_ci95_low"] = max(0.0, mean - half_width)
                row[f"{col}_ci95_high"] = mean + half_width
            else:
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan

            if row["total_low_hits"] == 0:
                row[f"{col}_comment"] = "zero low-tail hits; below current resolution"
            else:
                row[f"{col}_comment"] = ""

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["a_second", "h"]).reset_index(drop=True)