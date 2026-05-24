import math
import numpy as np
import pandas as pd


def normal_ci(mean, se, level=0.95):
    """
    Normal confidence interval.

    Parameters
    ----------
    mean : float
        Estimator value.
    se : float
        Standard error.
    level : float
        Confidence level. Only 0.95 is used here.

    Returns
    -------
    tuple
        Lower and upper confidence bounds.
    """
    if level != 0.95:
        raise NotImplementedError("Only 95% confidence intervals are implemented.")
    z = 1.96
    return mean - z * se, mean + z * se


def relative_error(mean, se):
    """
    Compute relative error SE / mean.
    """
    if mean <= 0:
        return np.inf
    return se / mean


def summarize_samples(samples, name="estimator"):
    """
    Summarize an array of Monte Carlo samples.

    Parameters
    ----------
    samples : array-like
        Samples whose expectation estimates the target probability.
    name : str
        Name of the estimator.

    Returns
    -------
    dict
        Summary statistics.
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    mean = x.mean()
    sample_var = x.var(ddof=1) if n > 1 else np.nan
    se = math.sqrt(sample_var / n) if n > 1 else np.nan
    ci_low, ci_high = normal_ci(mean, se)

    return {
        "method": name,
        "n": n,
        "estimate": mean,
        "sample_variance": sample_var,
        "standard_error": se,
        "relative_error": relative_error(mean, se),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def simulate_two_queues_exact(
    q_ask0,
    q_bid0,
    rates,
    rng,
    max_events=10_000_000,
    store_path=False,
):
    """
    Exact simulation of two independent birth-death queues until one hits zero.

    Parameters
    ----------
    q_ask0 : int
        Initial ask queue size.
    q_bid0 : int
        Initial bid queue size.
    rates : dict
        Dictionary with keys:
        - lambda_ask_plus
        - lambda_ask_minus
        - lambda_bid_plus
        - lambda_bid_minus
    rng : np.random.Generator
        Random number generator.
    max_events : int
        Safety cap on the number of events.
    store_path : bool
        Whether to store the full path.

    Returns
    -------
    dict
        Simulation output containing:
        - first_hit: "ask" or "bid"
        - tau: stopping time
        - q_ask: final ask size
        - q_bid: final bid size
        - n_events: number of events
        - path: optional path information
    """
    q_ask = int(q_ask0)
    q_bid = int(q_bid0)
    t = 0.0

    la_p = rates["lambda_ask_plus"]
    la_m = rates["lambda_ask_minus"]
    lb_p = rates["lambda_bid_plus"]
    lb_m = rates["lambda_bid_minus"]

    if min(la_p, la_m, lb_p, lb_m) < 0:
        raise ValueError("All intensities must be non-negative.")

    total_rate = la_p + la_m + lb_p + lb_m
    if total_rate <= 0:
        raise ValueError("Total rate must be positive.")

    if store_path:
        times = [t]
        ask_path = [q_ask]
        bid_path = [q_bid]
        events = []

    for n_events in range(1, max_events + 1):
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        u = rng.random() * total_rate

        if u < la_p:
            q_ask += 1
            event = "ask_plus"
        elif u < la_p + la_m:
            q_ask -= 1
            event = "ask_minus"
        elif u < la_p + la_m + lb_p:
            q_bid += 1
            event = "bid_plus"
        else:
            q_bid -= 1
            event = "bid_minus"

        if store_path:
            times.append(t)
            ask_path.append(q_ask)
            bid_path.append(q_bid)
            events.append(event)

        if q_ask <= 0:
            out = {
                "first_hit": "ask",
                "tau": t,
                "q_ask": q_ask,
                "q_bid": q_bid,
                "n_events": n_events,
            }
            if store_path:
                out["path"] = {
                    "times": np.array(times),
                    "ask": np.array(ask_path),
                    "bid": np.array(bid_path),
                    "events": np.array(events),
                }
            return out

        if q_bid <= 0:
            out = {
                "first_hit": "bid",
                "tau": t,
                "q_ask": q_ask,
                "q_bid": q_bid,
                "n_events": n_events,
            }
            if store_path:
                out["path"] = {
                    "times": np.array(times),
                    "ask": np.array(ask_path),
                    "bid": np.array(bid_path),
                    "events": np.array(events),
                }
            return out

    raise RuntimeError("max_events reached before either queue hit zero.")


def naive_monte_carlo(
    n_samples,
    q_ask0,
    q_bid0,
    rates,
    seed=123,
):
    """
    Estimate P(tau_ask < tau_bid) by naive Monte Carlo.

    Parameters
    ----------
    n_samples : int
        Number of independent simulations.
    q_ask0 : int
        Initial ask queue size.
    q_bid0 : int
        Initial bid queue size.
    rates : dict
        Original model intensities.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        summary, raw_samples
    """
    rng = np.random.default_rng(seed)

    indicators = np.empty(n_samples, dtype=float)
    taus = np.empty(n_samples, dtype=float)
    n_events_arr = np.empty(n_samples, dtype=int)

    for k in range(n_samples):
        out = simulate_two_queues_exact(
            q_ask0=q_ask0,
            q_bid0=q_bid0,
            rates=rates,
            rng=rng,
        )
        indicators[k] = 1.0 if out["first_hit"] == "ask" else 0.0
        taus[k] = out["tau"]
        n_events_arr[k] = out["n_events"]

    summary = summarize_samples(indicators, name="Naive Monte Carlo")
    summary["rare_events_observed"] = int(indicators.sum())
    summary["mean_tau"] = taus.mean()
    summary["mean_n_events"] = n_events_arr.mean()

    raw = {
        "indicators": indicators,
        "taus": taus,
        "n_events": n_events_arr,
    }

    return summary, raw


def simulate_until_ask_level_or_bid_zero(
    q_ask_start,
    q_bid_start,
    next_ask_level,
    rates,
    rng,
    max_events=10_000_000,
):
    """
    Simulate from a given state until either:

    - ask queue reaches next_ask_level or below,
    - bid queue reaches 0.

    Parameters
    ----------
    q_ask_start : int
        Starting ask queue size.
    q_bid_start : int
        Starting bid queue size.
    next_ask_level : int
        Target ask level.
    rates : dict
        Original rates.
    rng : np.random.Generator
        Random number generator.
    max_events : int
        Safety cap.

    Returns
    -------
    dict
        - success: True if ask reached next level before bid hit zero.
        - q_ask: final ask queue size.
        - q_bid: final bid queue size.
        - tau: simulated time duration.
        - n_events: number of events.
    """
    q_ask = int(q_ask_start)
    q_bid = int(q_bid_start)
    t = 0.0

    if q_ask <= next_ask_level:
        return {
            "success": True,
            "q_ask": q_ask,
            "q_bid": q_bid,
            "tau": t,
            "n_events": 0,
        }

    if q_bid <= 0:
        return {
            "success": False,
            "q_ask": q_ask,
            "q_bid": q_bid,
            "tau": t,
            "n_events": 0,
        }

    la_p = rates["lambda_ask_plus"]
    la_m = rates["lambda_ask_minus"]
    lb_p = rates["lambda_bid_plus"]
    lb_m = rates["lambda_bid_minus"]

    total_rate = la_p + la_m + lb_p + lb_m

    for n_events in range(1, max_events + 1):
        dt = rng.exponential(1.0 / total_rate)
        t += dt

        u = rng.random() * total_rate

        if u < la_p:
            q_ask += 1
        elif u < la_p + la_m:
            q_ask -= 1
        elif u < la_p + la_m + lb_p:
            q_bid += 1
        else:
            q_bid -= 1

        if q_ask <= next_ask_level:
            return {
                "success": True,
                "q_ask": q_ask,
                "q_bid": q_bid,
                "tau": t,
                "n_events": n_events,
            }

        if q_bid <= 0:
            return {
                "success": False,
                "q_ask": q_ask,
                "q_bid": q_bid,
                "tau": t,
                "n_events": n_events,
            }

    raise RuntimeError("max_events reached.")


def fixed_multilevel_splitting(
    n_particles,
    levels,
    q_ask0,
    q_bid0,
    rates,
    seed=123,
):
    """
    Fixed Multilevel Splitting estimator.

    Parameters
    ----------
    n_particles : int
        Number of particles used at each level.
    levels : list or array
        Decreasing ask levels, including q_ask0 and 0.
        Example: [80, 70, 60, 50, 40, 30, 20, 10, 0]
    q_ask0 : int
        Initial ask queue size.
    q_bid0 : int
        Initial bid queue size.
    rates : dict
        Original intensities.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Summary of the splitting estimator.
    """
    rng = np.random.default_rng(seed)

    levels = list(levels)
    if levels[0] != q_ask0:
        raise ValueError("The first level must be q_ask0.")
    if levels[-1] != 0:
        raise ValueError("The final level must be 0.")
    if any(levels[i] <= levels[i + 1] for i in range(len(levels) - 1)):
        raise ValueError("Levels must be strictly decreasing.")

    # Initial particles are all at the initial state.
    particles = [(q_ask0, q_bid0) for _ in range(n_particles)]

    conditional_probabilities = []
    survivors_by_level = []
    mean_bid_by_level = []
    mean_ask_by_level = []

    total_events = 0

    for k in range(1, len(levels)):
        next_level = levels[k]

        survivors = []

        for q_ask_start, q_bid_start in particles:
            out = simulate_until_ask_level_or_bid_zero(
                q_ask_start=q_ask_start,
                q_bid_start=q_bid_start,
                next_ask_level=next_level,
                rates=rates,
                rng=rng,
            )
            total_events += out["n_events"]

            if out["success"]:
                survivors.append((out["q_ask"], out["q_bid"]))

        n_success = len(survivors)
        p_cond = n_success / n_particles

        conditional_probabilities.append(p_cond)
        survivors_by_level.append(n_success)

        if n_success == 0:
            estimate = 0.0
            return {
                "method": "Fixed Multilevel Splitting",
                "estimate": estimate,
                "n_particles": n_particles,
                "levels": levels,
                "conditional_probabilities": conditional_probabilities,
                "survivors_by_level": survivors_by_level,
                "total_events": total_events,
                "failed": True,
            }

        survivor_array = np.array(survivors)
        mean_ask_by_level.append(survivor_array[:, 0].mean())
        mean_bid_by_level.append(survivor_array[:, 1].mean())

        # Resample/cloning step: sample N particles with replacement from survivors.
        indices = rng.integers(0, n_success, size=n_particles)
        particles = [survivors[i] for i in indices]

    estimate = float(np.prod(conditional_probabilities))

    return {
        "method": "Fixed Multilevel Splitting",
        "estimate": estimate,
        "n_particles": n_particles,
        "levels": levels,
        "conditional_probabilities": conditional_probabilities,
        "survivors_by_level": survivors_by_level,
        "mean_ask_by_level": mean_ask_by_level,
        "mean_bid_by_level": mean_bid_by_level,
        "total_events": total_events,
        "failed": False,
    }


def repeat_fixed_splitting(
    n_repeats,
    n_particles,
    levels,
    q_ask0,
    q_bid0,
    rates,
    seed=123,
):
    """
    Repeat fixed multilevel splitting several times to estimate variability.

    Returns
    -------
    tuple
        summary, dataframe of repetitions
    """
    rows = []

    for r in range(n_repeats):
        out = fixed_multilevel_splitting(
            n_particles=n_particles,
            levels=levels,
            q_ask0=q_ask0,
            q_bid0=q_bid0,
            rates=rates,
            seed=seed + r,
        )

        rows.append({
            "repeat": r,
            "estimate": out["estimate"],
            "failed": out["failed"],
            "total_events": out["total_events"],
        })

    df = pd.DataFrame(rows)

    summary = summarize_samples(df["estimate"].values, name="Fixed Multilevel Splitting")
    summary["n_repeats"] = n_repeats
    summary["n_particles"] = n_particles
    summary["failed_repeats"] = int(df["failed"].sum())
    summary["mean_total_events"] = df["total_events"].mean()

    return summary, df
