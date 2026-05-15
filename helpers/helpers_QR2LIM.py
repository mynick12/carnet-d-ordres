import math
import numpy as np
from statistics import NormalDist

from .helpers_QRNIID import *

#Useful parameters
lambda_floor = 1e-8

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


def simulate_reactive_second_limit_until_time(
    q2_0,
    horizon,
    driver_removal_times,
    mu_second_add,
    lambda_second_remove,
    a,
    b,
    rng=None,
    max_events=2_000_000,
):
    """
    Simule une deuxième limite dont l'intensité d'ajout est augmentée
    par les retraits de la première limite du même côté.

    lambda_2_plus(t) = mu_second_add + Z(t)
    dZ(t) = -b Z(t) dt + a dN_{1,-}(t)

    Les temps driver_removal_times sont les instants de retrait de la première limite.
    """
    if rng is None:
        rng = np.random.default_rng()

    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    driver_removal_times = np.asarray(driver_removal_times, dtype=float)
    driver_removal_times = np.sort(driver_removal_times[driver_removal_times <= horizon])

    t = 0.0
    q = int(q2_0)
    z = 0.0
    driver_idx = 0

    for _ in range(max_events):
        if t >= horizon:
            return q

        next_driver_time = (
            driver_removal_times[driver_idx]
            if driver_idx < len(driver_removal_times)
            else np.inf
        )

        active_remove = lambda_second_remove if q > 0 else 0.0

        # Sur l'intervalle avant le prochain retrait de la première limite,
        # z décroît, donc l'intensité maximale est l'intensité courante.
        upper_rate = mu_second_add + z + active_remove

        if upper_rate <= 0:
            t_next = min(next_driver_time, horizon)
            if np.isfinite(t_next):
                z *= np.exp(-b * (t_next - t))
            t = t_next

            if t == next_driver_time:
                z += a
                driver_idx += 1
                continue

            return q

        dt = rng.exponential(1.0 / upper_rate)
        candidate_time = t + dt

        # Si un retrait de la première limite arrive avant le prochain événement de N2,
        # on avance jusqu'à ce temps, on fait décroître z, puis on ajoute l'impulsion a.
        if next_driver_time <= candidate_time and next_driver_time <= horizon:
            z *= np.exp(-b * (next_driver_time - t))
            t = next_driver_time
            z += a
            driver_idx += 1
            continue

        # Si l'horizon arrive avant le prochain événement de N2
        if candidate_time > horizon:
            z *= np.exp(-b * (horizon - t))
            t = horizon
            return q

        # Sinon, on teste un événement candidat de N2
        z_candidate = z * np.exp(-b * dt)
        active_remove = lambda_second_remove if q > 0 else 0.0
        true_rate = mu_second_add + z_candidate + active_remove

        # Amincissement
        if rng.random() > true_rate / upper_rate:
            t = candidate_time
            z = z_candidate
            continue

        t = candidate_time
        z = z_candidate

        # Type d'événement sur la deuxième limite
        if rng.random() < (mu_second_add + z) / true_rate:
            q += 1
        else:
            q -= 1

    raise RuntimeError("max_events reached before the simulation finished")


def sample_second_limit_conditionals_reactive(
    n_samples,
    q1_0,
    q2_0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    lambda_second_add,
    lambda_second_remove,
    a_second,
    b_second,
    gamma_cross=1.0,
    rng=None,
    t_max=10_000.0,
):
    """
    Question 1.2.5.2 :
    simule N^{+2}_tau avec une dynamique réactive.

    N^{+2} est renforcée par les retraits sur N^{+1}.
    """
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
            # Comme bid correspond à +1, les retraits de +1 sont bid_removal_times.
            driver_times = first_limit["bid_removal_times"]

            plus2_at_tau[k] = simulate_reactive_second_limit_until_time(
                q2_0=q2_0,
                horizon=tau_first[k],
                driver_removal_times=driver_times,
                mu_second_add=lambda_second_add,
                lambda_second_remove=lambda_second_remove,
                a=a_second,
                b=b_second,
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

