import copy
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

LAMBDA_FLOOR = 1e-8


def mean_confidence_interval(x: Iterable[float], level: float = 0.95) -> Tuple[float, float, float]:
    """Normal confidence interval for the mean of repeated estimates."""
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), np.nan, np.nan

    mean = float(x.mean())
    s = float(x.std(ddof=1))
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    half_width = z * s / math.sqrt(len(x))
    return mean, mean - half_width, mean + half_width


def binomial_confidence_interval(p_hat: float, n: int, level: float = 0.95) -> Tuple[float, float, float]:
    """Normal approximation confidence interval for a binomial probability."""
    if n <= 0 or not np.isfinite(p_hat):
        return np.nan, np.nan, np.nan

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)
    return p_hat - z * se, p_hat + z * se, se


def safe_relative_error(se: float, estimate: float) -> float:
    if estimate == 0 or not np.isfinite(estimate) or not np.isfinite(se):
        return np.inf
    return abs(se / estimate)


# ============================================================
# Model containers
# ============================================================

@dataclass
class ModelParams2LIM:
    """Parameters of the 2LIM model from Question 1.2.5.1."""

    q1_0: int = 10
    q2_0: int = 10

    # First limits: coupled Hawkes dynamics
    mu_plus: float = 1.0
    mu_minus: float = 1.1
    alpha: float = 0.50
    beta: float = 0.50
    gamma_cross: float = 1.0
    lambda_floor: float = LAMBDA_FLOOR

    # Second limit: constant birth-death dynamics
    lambda_second_add: float = 1.500
    lambda_second_remove: float = 0.500

    # Rare threshold
    h: int = 10

    # Safety
    max_events: int = 2_000_000


@dataclass
class Particle2LIM:
    """State of one splitting particle."""

    t: float
    q_plus1: int
    q_minus1: int
    q_plus2: int
    memory_plus1: float
    memory_minus1: float
    n_events: int
    status: str = "active"  # active or rare_success


@dataclass
class FMSSettings2LIM:
    """Settings for the improved Fixed Multilevel Splitting algorithm."""

    n_particles: int = 2_000
    n_repeats: int = 20
    score_levels: Tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    q2_margin: int = 20
    w_q2: float = 0.25
    seed: int = 12345


# ============================================================
# Model dynamics
# ============================================================


def initial_particle(params: ModelParams2LIM) -> Particle2LIM:
    return Particle2LIM(
        t=0.0,
        q_plus1=int(params.q1_0),
        q_minus1=int(params.q1_0),
        q_plus2=int(params.q2_0),
        memory_plus1=0.0,
        memory_minus1=0.0,
        n_events=0,
        status="active",
    )


def two_queue_intensities(
    memory_plus1: float,
    memory_minus1: float,
    params: ModelParams2LIM,
) -> Dict[str, float]:
    """
    Intensities for the two first limits.

    plus1 corresponds to N^{+1}; minus1 corresponds to N^{-1}.
    Additions have constant rate mu_plus. Removals have Hawkes-type coupled intensities.
    """
    plus1_signal = memory_plus1 + params.gamma_cross * memory_minus1
    minus1_signal = memory_minus1 + params.gamma_cross * memory_plus1

    return {
        "plus1_plus": params.mu_plus,
        "plus1_minus": max(params.mu_minus + plus1_signal, params.lambda_floor),
        "minus1_plus": params.mu_plus,
        "minus1_minus": max(params.mu_minus + minus1_signal, params.lambda_floor),
    }


def total_rate_upper_bound(p: Particle2LIM, params: ModelParams2LIM) -> float:
    """
    Upper bound used for thinning.

    Between events, positive memory decays downward and negative memory decays upward,
    hence the maximal removal intensity before the next event is bounded by the positive
    part of the current signal plus the baseline.
    """
    plus1_signal = p.memory_plus1 + params.gamma_cross * p.memory_minus1
    minus1_signal = p.memory_minus1 + params.gamma_cross * p.memory_plus1

    upper_plus1_minus = max(params.mu_minus + max(plus1_signal, 0.0), params.mu_minus, params.lambda_floor)
    upper_minus1_minus = max(params.mu_minus + max(minus1_signal, 0.0), params.mu_minus, params.lambda_floor)

    active_second_remove = params.lambda_second_remove if p.q_plus2 > 0 else 0.0

    return (
        params.mu_plus
        + upper_plus1_minus
        + params.mu_plus
        + upper_minus1_minus
        + params.lambda_second_add
        + active_second_remove
    )


def rare_event_success(p: Particle2LIM, params: ModelParams2LIM) -> bool:
    """Target event: tau = tau_{+1} and N^{+2}_tau <= h."""
    return p.q_plus1 <= 0 and p.q_minus1 > 0 and p.q_plus2 <= params.h


def tau_plus1_success(p: Particle2LIM) -> bool:
    """Conditioning event: tau = tau_{+1}."""
    return p.q_plus1 <= 0 and p.q_minus1 > 0


def is_terminal(p: Particle2LIM) -> bool:
    """Terminal event: one of the first limits has reached zero."""
    return p.q_plus1 <= 0 or p.q_minus1 <= 0


def step_particle(p: Particle2LIM, params: ModelParams2LIM, rng: np.random.Generator) -> Particle2LIM:
    """Simulate one accepted event of the full process using thinning."""
    p = copy.deepcopy(p)

    while True:
        upper_rate = total_rate_upper_bound(p, params)
        dt = rng.exponential(1.0 / upper_rate)

        decay = math.exp(-params.beta * dt)
        candidate_memory_plus1 = p.memory_plus1 * decay
        candidate_memory_minus1 = p.memory_minus1 * decay

        first_rates = two_queue_intensities(
            memory_plus1=candidate_memory_plus1,
            memory_minus1=candidate_memory_minus1,
            params=params,
        )

        active_second_remove = params.lambda_second_remove if p.q_plus2 > 0 else 0.0

        rates = {
            "plus1_plus": first_rates["plus1_plus"],
            "plus1_minus": first_rates["plus1_minus"],
            "minus1_plus": first_rates["minus1_plus"],
            "minus1_minus": first_rates["minus1_minus"],
            "plus2_plus": params.lambda_second_add,
            "plus2_minus": active_second_remove,
        }

        true_rate = sum(rates.values())

        # Rejection step of thinning.
        if rng.random() > true_rate / upper_rate:
            p.t += dt
            p.memory_plus1 = candidate_memory_plus1
            p.memory_minus1 = candidate_memory_minus1
            continue

        # Accepted event.
        p.t += dt
        p.memory_plus1 = candidate_memory_plus1
        p.memory_minus1 = candidate_memory_minus1
        p.n_events += 1

        u = rng.random() * true_rate
        cumulative = 0.0

        cumulative += rates["plus1_plus"]
        if u < cumulative:
            p.q_plus1 += 1
            p.memory_plus1 += params.alpha
            return p

        cumulative += rates["plus1_minus"]
        if u < cumulative:
            p.q_plus1 -= 1
            p.memory_plus1 -= params.alpha
            return p

        cumulative += rates["minus1_plus"]
        if u < cumulative:
            p.q_minus1 += 1
            p.memory_minus1 += params.alpha
            return p

        cumulative += rates["minus1_minus"]
        if u < cumulative:
            p.q_minus1 -= 1
            p.memory_minus1 -= params.alpha
            return p

        cumulative += rates["plus2_plus"]
        if u < cumulative:
            p.q_plus2 += 1
            return p

        # plus2 removal; inactive at zero by construction.
        if p.q_plus2 > 0:
            p.q_plus2 -= 1
        return p


def simulate_until_terminal(
    particle: Particle2LIM,
    params: ModelParams2LIM,
    rng: np.random.Generator,
) -> Tuple[Particle2LIM, Dict[str, object]]:
    """Simulate until one of the two first limits hits zero."""
    p = copy.deepcopy(particle)

    if p.status == "rare_success":
        return p, {"reason": "already_rare_success", "tau": p.t, "q_plus2": p.q_plus2}

    for _ in range(params.max_events):
        if is_terminal(p):
            reason = "plus1_hit_zero" if tau_plus1_success(p) else "minus1_hit_zero"
            if rare_event_success(p, params):
                p.status = "rare_success"
            return p, {"reason": reason, "tau": p.t, "q_plus2": p.q_plus2}

        p = step_particle(p, params, rng)

    raise RuntimeError("max_events reached before terminal event")


# ============================================================
# Score for improved splitting
# ============================================================


def splitting_score(
    p: Particle2LIM,
    params: ModelParams2LIM,
    q2_margin: int = 20,
    w_q2: float = 0.25,
) -> float:
    """
    Combined score for Fixed Multilevel Splitting.

    The score has two terms:
    1. progress of N^{+1} toward zero: q1_0 - q_plus1;
    2. bonus for small N^{+2}: max(0, h + q2_margin - q_plus2).

    The second term is weighted by w_q2. This makes particles with low N^{+2}
    more likely to survive near the final levels.
    """
    progress_plus1 = params.q1_0 - p.q_plus1
    small_q2_bonus = max(0.0, params.h + q2_margin - p.q_plus2)
    return progress_plus1 + w_q2 * small_q2_bonus


def simulate_until_score_or_terminal(
    particle: Particle2LIM,
    score_level: float,
    params: ModelParams2LIM,
    rng: np.random.Generator,
    q2_margin: int = 20,
    w_q2: float = 0.25,
) -> Tuple[bool, Optional[Particle2LIM], Dict[str, object]]:
    """
    Simulate one particle until it reaches a score level or until terminal failure/success.

    Returns success=True if the particle either reaches the intermediate score level
    or hits the rare event before reaching the level. Rare successes are treated as
    absorbing successful particles for the remaining splitting levels.
    """
    p = copy.deepcopy(particle)

    if p.status == "rare_success":
        return True, p, {
            "reason": "already_rare_success",
            "tau": p.t,
            "q_plus2": p.q_plus2,
            "score": splitting_score(p, params, q2_margin, w_q2),
        }

    for _ in range(params.max_events):
        current_score = splitting_score(p, params, q2_margin, w_q2)

        if current_score >= score_level:
            return True, p, {
                "reason": "reached_score_level",
                "tau": p.t,
                "q_plus2": p.q_plus2,
                "score": current_score,
            }

        if is_terminal(p):
            if rare_event_success(p, params):
                p.status = "rare_success"
                return True, p, {
                    "reason": "rare_success_before_level",
                    "tau": p.t,
                    "q_plus2": p.q_plus2,
                    "score": current_score,
                }

            reason = "terminal_plus1_but_q2_too_large" if tau_plus1_success(p) else "minus1_hit_zero"
            return False, None, {
                "reason": reason,
                "tau": p.t,
                "q_plus2": p.q_plus2,
                "score": current_score,
            }

        p = step_particle(p, params, rng)

    raise RuntimeError("max_events reached before score level or terminal event")


# ============================================================
# Fixed Multilevel Splitting estimators
# ============================================================


def fixed_multilevel_splitting_combined_score(
    params: ModelParams2LIM,
    settings: FMSSettings2LIM,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Improved Fixed Multilevel Splitting estimator of the numerator:

        P(N^{+2}_tau <= h and tau = tau_{+1}).

    It uses a combined score that favours both progress of N^{+1} toward zero
    and small values of N^{+2}.
    """
    if seed is None:
        seed = settings.seed

    rng = np.random.default_rng(seed)
    particles: List[Particle2LIM] = [initial_particle(params) for _ in range(settings.n_particles)]

    conditional_probabilities: List[float] = []
    survivors_by_level: List[int] = []
    level_infos: List[Dict[str, object]] = []

    for score_level in settings.score_levels:
        survivors: List[Particle2LIM] = []
        infos: List[Dict[str, object]] = []

        for particle in particles:
            success, new_particle, info = simulate_until_score_or_terminal(
                particle=particle,
                score_level=score_level,
                params=params,
                rng=rng,
                q2_margin=settings.q2_margin,
                w_q2=settings.w_q2,
            )
            infos.append(info)
            if success:
                survivors.append(new_particle)

        n_survivors = len(survivors)
        p_cond = n_survivors / settings.n_particles

        conditional_probabilities.append(p_cond)
        survivors_by_level.append(n_survivors)

        if n_survivors == 0:
            return {
                "method": "FMS combined score numerator",
                "estimate": 0.0,
                "failed": True,
                "failed_at_score_level": score_level,
                "params": params,
                "settings": settings,
                "conditional_probabilities": conditional_probabilities,
                "survivors_by_level": survivors_by_level,
                "level_infos": level_infos,
            }

        q2_values = np.array([p.q_plus2 for p in survivors], dtype=float)
        q1_values = np.array([p.q_plus1 for p in survivors], dtype=float)
        tau_values = np.array([p.t for p in survivors], dtype=float)
        scores = np.array([splitting_score(p, params, settings.q2_margin, settings.w_q2) for p in survivors])
        rare_absorbed = sum(p.status == "rare_success" for p in survivors)

        reasons = pd.Series([info["reason"] for info in infos]).value_counts().to_dict()

        level_infos.append({
            "score_level": score_level,
            "n_survivors": n_survivors,
            "p_cond": p_cond,
            "rare_absorbed": rare_absorbed,
            "mean_q_plus1": float(q1_values.mean()),
            "mean_q_plus2": float(q2_values.mean()),
            "min_q_plus2": float(q2_values.min()),
            "max_q_plus2": float(q2_values.max()),
            "mean_tau": float(tau_values.mean()),
            "mean_score": float(scores.mean()),
            "reasons": reasons,
        })

        # Resample successful particles with replacement.
        indices = rng.integers(0, n_survivors, size=settings.n_particles)
        particles = [copy.deepcopy(survivors[i]) for i in indices]

    # Final stage: from the last score level to the true rare event.
    final_successes = 0
    final_particles: List[Particle2LIM] = []
    final_infos: List[Dict[str, object]] = []

    for particle in particles:
        if particle.status == "rare_success":
            final_successes += 1
            final_particles.append(particle)
            final_infos.append({
                "reason": "already_rare_success",
                "tau": particle.t,
                "q_plus2": particle.q_plus2,
            })
            continue

        terminal_particle, info = simulate_until_terminal(particle, params, rng)
        final_particles.append(terminal_particle)
        final_infos.append(info)

        if rare_event_success(terminal_particle, params):
            final_successes += 1

    p_final = final_successes / settings.n_particles
    conditional_probabilities.append(p_final)
    survivors_by_level.append(final_successes)

    estimate = float(np.prod(conditional_probabilities))

    final_q2 = np.array([p.q_plus2 for p in final_particles], dtype=float)
    final_reasons = pd.Series([info["reason"] for info in final_infos]).value_counts().to_dict()

    level_infos.append({
        "score_level": "final_event",
        "n_survivors": final_successes,
        "p_cond": p_final,
        "rare_absorbed": sum(p.status == "rare_success" for p in final_particles),
        "mean_q_plus1": float(np.mean([p.q_plus1 for p in final_particles])),
        "mean_q_plus2": float(final_q2.mean()),
        "min_q_plus2": float(final_q2.min()),
        "max_q_plus2": float(final_q2.max()),
        "mean_tau": float(np.mean([p.t for p in final_particles])),
        "mean_score": float(np.mean([splitting_score(p, params, settings.q2_margin, settings.w_q2) for p in final_particles])),
        "reasons": final_reasons,
    })

    return {
        "method": "FMS combined score numerator",
        "estimate": estimate,
        "failed": False,
        "failed_at_score_level": None,
        "params": params,
        "settings": settings,
        "conditional_probabilities": conditional_probabilities,
        "survivors_by_level": survivors_by_level,
        "level_infos": level_infos,
    }


def repeat_fms_combined_score(
    params: ModelParams2LIM,
    settings: FMSSettings2LIM,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Repeat the improved FMS numerator estimator and summarize its variance."""
    rows = []

    for r in range(settings.n_repeats):
        out = fixed_multilevel_splitting_combined_score(
            params=params,
            settings=settings,
            seed=settings.seed + 10_000 + r,
        )

        rows.append({
            "repeat": r,
            "numerator_estimate": out["estimate"],
            "failed": out["failed"],
            "failed_at_score_level": out["failed_at_score_level"],
        })

    repeats = pd.DataFrame(rows)
    x = repeats.loc[~repeats["failed"], "numerator_estimate"].to_numpy(dtype=float)

    mean, ci_low, ci_high = mean_confidence_interval(x)
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan

    summary = {
        "method": "Repeated FMS combined score numerator",
        "n_repeats": settings.n_repeats,
        "n_particles": settings.n_particles,
        "score_levels": settings.score_levels,
        "q2_margin": settings.q2_margin,
        "w_q2": settings.w_q2,
        "numerator_estimate": mean,
        "standard_error": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "relative_error": safe_relative_error(se, mean),
        "n_valid_repeats": int(len(x)),
        "n_failed_repeats": int(repeats["failed"].sum()),
    }

    return summary, repeats


# ============================================================
# Naive Monte Carlo estimators
# ============================================================


def simulate_one_path(params: ModelParams2LIM, rng: np.random.Generator) -> Dict[str, object]:
    """Simulate one complete path until one first limit hits zero."""
    p0 = initial_particle(params)
    p_terminal, info = simulate_until_terminal(p0, params, rng)

    return {
        "tau": p_terminal.t,
        "q_plus1_tau": p_terminal.q_plus1,
        "q_minus1_tau": p_terminal.q_minus1,
        "q_plus2_tau": p_terminal.q_plus2,
        "tau_plus1": tau_plus1_success(p_terminal),
        "rare_event": rare_event_success(p_terminal, params),
        "reason": info["reason"],
    }


def naive_monte_carlo_conditional(
    params: ModelParams2LIM,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Dict[str, object]:
    """Naive estimator of P(N^{+2}_tau <= h | tau = tau_{+1})."""
    rng = np.random.default_rng(seed)

    conditioned = 0
    rare = 0
    q2_conditioned = []
    tau_conditioned = []

    for _ in range(n_samples):
        out = simulate_one_path(params, rng)
        if out["tau_plus1"]:
            conditioned += 1
            q2_conditioned.append(out["q_plus2_tau"])
            tau_conditioned.append(out["tau"])
            if out["rare_event"]:
                rare += 1

    p_hat = rare / conditioned if conditioned > 0 else np.nan
    ci_low, ci_high, se = binomial_confidence_interval(p_hat, conditioned)

    return {
        "method": "Naive conditional Monte Carlo",
        "n_samples": n_samples,
        "conditioned_samples": conditioned,
        "rare_events": rare,
        "p_tau_plus1": conditioned / n_samples,
        "estimate": p_hat,
        "standard_error": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "relative_error": safe_relative_error(se, p_hat),
        "mean_q2_cond": float(np.mean(q2_conditioned)) if conditioned > 0 else np.nan,
        "mean_tau_cond": float(np.mean(tau_conditioned)) if conditioned > 0 else np.nan,
    }


def estimate_denominator_tau_plus1(
    params: ModelParams2LIM,
    n_samples: int = 100_000,
    seed: int = 12345,
) -> Dict[str, object]:
    """Naive estimator of P(tau = tau_{+1}), which is not rare in this symmetric setting."""
    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_samples):
        out = simulate_one_path(params, rng)
        count += int(out["tau_plus1"])

    p_hat = count / n_samples
    ci_low, ci_high, se = binomial_confidence_interval(p_hat, n_samples)

    return {
        "method": "Naive Monte Carlo denominator",
        "n_samples": n_samples,
        "tau_plus1_count": count,
        "estimate": p_hat,
        "standard_error": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "relative_error": safe_relative_error(se, p_hat),
    }


def conditional_from_numerator_and_denominator(
    numerator_summary: Dict[str, object],
    denominator_summary: Dict[str, object],
) -> Dict[str, object]:
    """
    Combine numerator estimate P(A and tau=tau_{+1}) with denominator estimate
    P(tau=tau_{+1}) to estimate the conditional probability.
    """
    numerator = numerator_summary["numerator_estimate"]
    denominator = denominator_summary["estimate"]
    conditional = numerator / denominator if denominator > 0 else np.nan

    # Delta-method approximation, assuming independent numerator and denominator estimators.
    se_num = numerator_summary.get("standard_error", np.nan)
    se_den = denominator_summary.get("standard_error", np.nan)

    if (
        np.isfinite(conditional)
        and np.isfinite(se_num)
        and np.isfinite(se_den)
        and numerator > 0
        and denominator > 0
    ):
        rel_var = (se_num / numerator) ** 2 + (se_den / denominator) ** 2
        se_cond = conditional * math.sqrt(rel_var)
        z = NormalDist().inv_cdf(0.975)
        ci_low = conditional - z * se_cond
        ci_high = conditional + z * se_cond
    else:
        se_cond = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        "method": "FMS numerator / MC denominator",
        "numerator_estimate": numerator,
        "denominator_estimate": denominator,
        "estimate": conditional,
        "standard_error": se_cond,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "relative_error": safe_relative_error(se_cond, conditional),
    }
