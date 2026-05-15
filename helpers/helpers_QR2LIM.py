import math
import numpy as np
from statistics import NormalDist

from helpers_QRNIID import *

def simulate_constant_queue_until_time(
    q0,
    lambda_add,
    lambda_remove,
    horizon,
    rng=None,
    max_events=2_000_000,
):
    """Exact simulation of a constant-rate queue on a fixed time interval."""
    if rng is None:
        rng = np.random.default_rng()
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if lambda_add < 0 or lambda_remove < 0:
        raise ValueError("intensities must be non-negative")

    t = 0.0
    q = int(q0)
    if q < 0:
        raise ValueError("queue size must be non-negative")

    for _ in range(max_events):
        active_remove = lambda_remove if q > 0 else 0.0
        rate = lambda_add + active_remove
        if rate <= 0.0:
            return q

        dt = rng.exponential(1.0 / rate)
        if t + dt > horizon:
            return q

        t += dt
        if rng.random() < lambda_add / rate:
            q += 1
        else:
            q -= 1

    raise RuntimeError("max_events reached before the fixed horizon")


def plus_minus_label_from_first_queue(first_queue):
    if first_queue == "bid":
        return "plus1"
    if first_queue == "ask":
        return "minus1"
    return first_queue


def empirical_cdf_distance(x, y):
    """Kolmogorov distance between two empirical integer distributions."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return np.nan

    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return np.max(np.abs(cdf_x - cdf_y))


def sample_second_limit_conditionals_constant(
    n_samples,
    q1_0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    gamma_cross=1.0,
    rng=None,
    t_max=10_000.0,
):
    if rng is None:
        rng = np.random.default_rng()

    plus2_at_tau = np.empty(n_samples, dtype=float)
    tau_first = np.empty(n_samples, dtype=float)
    first_empty = np.empty(n_samples, dtype=object)
    hit_zero = np.empty(n_samples, dtype=bool)

    for k in range(n_samples):
        first_limit = simulate_two_coupled_hawkes_queues(
            q_bid0=q1_0,
            q_ask0=q1_0,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            beta=beta,
            gamma_cross=gamma_cross,
            rng=rng,
            t_max=t_max,
            stop_at_zero=True,
            lambda_floor=lambda_floor,
        )
        tau_first[k] = first_limit["tau_first"]
        first_empty[k] = plus_minus_label_from_first_queue(first_limit["first_queue"])
        hit_zero[k] = first_limit["hit_zero"]

        if hit_zero[k]:
            plus2_at_tau[k] = simulate_constant_queue_until_time(
                q2_0,
                lambda_second_add,
                lambda_second_remove,
                tau_first[k],
                rng=rng,
            )
        else:
            plus2_at_tau[k] = np.nan

    return {
        "plus2_at_tau": plus2_at_tau,
        "tau_first": tau_first,
        "first_empty": first_empty,
        "hit_zero": hit_zero,
    }