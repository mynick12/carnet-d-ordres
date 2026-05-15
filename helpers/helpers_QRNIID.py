import math
import numpy as np
from statistics import NormalDist

#Useful parameters
lambda_floor = 1e-8

def lambda_minus_from_memory(memory, mu_minus, lambda_floor=lambda_floor):
    return max(mu_minus + memory, lambda_floor)


def total_rate_upper_bound(memory, mu_plus, mu_minus, lambda_floor=lambda_floor):
    # Between events, positive memory decays downward and negative memory decays upward.
    # Hence the largest possible removal intensity before the next event is either
    # the current value, if memory > 0, or the baseline mu_minus, if memory < 0.
    upper_lambda_minus = max(mu_minus + max(memory, 0.0), mu_minus, lambda_floor)
    return mu_plus + upper_lambda_minus


def simulate_one_hawkes_queue(
    q0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    rng=None,
    t_max=10_000.0,
    stop_at_zero=True,
    lambda_floor=lambda_floor,
    max_events=1_000_000,
):
    """Simulate one queue with constant arrivals and Hawkes-type removals."""
    if rng is None:
        rng = np.random.default_rng()

    t = 0.0
    q = int(q0)
    memory = 0.0

    times = [t]
    queue = [q]
    lambda_minus_path = [lambda_minus_from_memory(memory, mu_minus, lambda_floor)]
    event_types = []
    arrival_times = []
    removal_times = []

    for _ in range(max_events):
        if stop_at_zero and q <= 0:
            break

        upper_rate = total_rate_upper_bound(memory, mu_plus, mu_minus, lambda_floor)
        dt = rng.exponential(1.0 / upper_rate)

        if t + dt > t_max:
            memory *= math.exp(-beta * (t_max - t))
            t = t_max
            times.append(t)
            queue.append(q)
            lambda_minus_path.append(lambda_minus_from_memory(memory, mu_minus, lambda_floor))
            break

        candidate_memory = memory * math.exp(-beta * dt)
        candidate_lambda_minus = lambda_minus_from_memory(candidate_memory, mu_minus, lambda_floor)
        candidate_total_rate = mu_plus + candidate_lambda_minus

        # Accept or reject the candidate event.
        if rng.random() > candidate_total_rate / upper_rate:
            t += dt
            memory = candidate_memory
            continue

        t += dt
        memory = candidate_memory

        if rng.random() < mu_plus / candidate_total_rate:
            q += 1
            memory += alpha
            event_types.append("arrival")
            arrival_times.append(t)
        else:
            q -= 1
            memory -= alpha
            event_types.append("removal")
            removal_times.append(t)

        times.append(t)
        queue.append(q)
        lambda_minus_path.append(lambda_minus_from_memory(memory, mu_minus, lambda_floor))

        if stop_at_zero and q <= 0:
            break
    else:
        raise RuntimeError("max_events reached before the simulation finished")

    tau_zero = t if q <= 0 else np.nan

    return {
        "times": np.asarray(times),
        "queue": np.asarray(queue, dtype=int),
        "lambda_minus": np.asarray(lambda_minus_path),
        "event_types": np.asarray(event_types),
        "arrival_times": np.asarray(arrival_times),
        "removal_times": np.asarray(removal_times),
        "tau_zero": tau_zero,
        "hit_zero": q <= 0,
    }


def estimate_stationary_removal_rates(
    n_paths,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    rng=None,
    t_max=1_000.0,
    burn_in=200.0,
):
    if rng is None:
        rng = np.random.default_rng()

    estimates = np.empty(n_paths)
    for k in range(n_paths):
        out = simulate_one_hawkes_queue(
            q0=0,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            beta=beta,
            rng=rng,
            t_max=t_max,
            stop_at_zero=False,
            lambda_floor=lambda_floor,
        )
        n_removals = np.sum(out["removal_times"] >= burn_in)
        estimates[k] = n_removals / (t_max - burn_in)

    return estimates


def sample_hawkes_hitting_times(
    n_samples,
    q0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    rng=None,
    t_max=10_000.0,
):
    if rng is None:
        rng = np.random.default_rng()

    tau = np.empty(n_samples)
    hit = np.empty(n_samples, dtype=bool)

    for k in range(n_samples):
        out = simulate_one_hawkes_queue(
            q0=q0,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            beta=beta,
            rng=rng,
            t_max=t_max,
            stop_at_zero=True,
            lambda_floor=lambda_floor,
        )
        tau[k] = out["tau_zero"]
        hit[k] = out["hit_zero"]

    return tau, hit


def mean_confidence_interval(x, level=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    s = x.std(ddof=1)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    half_width = z * s / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


def theoretical_stationary_lambda_minus(mu_plus, mu_minus, alpha, beta):
    eta = alpha / beta
    return (mu_minus + eta * mu_plus) / (1.0 + eta)


def coupled_stationary_lambda_minus(mu_plus, mu_minus, alpha, beta, gamma_cross=1.0):
    eta = alpha / beta
    influence = (1.0 + gamma_cross) * eta
    return (mu_minus + influence * mu_plus) / (1.0 + influence)


def two_queue_intensities(memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross=1.0, lambda_floor=lambda_floor):
    bid_minus_signal = memory_bid + gamma_cross * memory_ask
    ask_minus_signal = memory_ask + gamma_cross * memory_bid

    lambda_bid_plus = mu_plus
    lambda_ask_plus = mu_plus
    lambda_bid_minus = max(mu_minus + bid_minus_signal, lambda_floor)
    lambda_ask_minus = max(mu_minus + ask_minus_signal, lambda_floor)

    return lambda_bid_plus, lambda_bid_minus, lambda_ask_plus, lambda_ask_minus


def two_queue_total_rate_upper_bound(memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross=1.0, lambda_floor=lambda_floor):
    bid_signal = memory_bid + gamma_cross * memory_ask
    ask_signal = memory_ask + gamma_cross * memory_bid

    upper_bid_minus = max(mu_minus + max(bid_signal, 0.0), mu_minus, lambda_floor)
    upper_ask_minus = max(mu_minus + max(ask_signal, 0.0), mu_minus, lambda_floor)

    return 2.0 * mu_plus + upper_bid_minus + upper_ask_minus


def simulate_two_coupled_hawkes_queues(
    q_bid0,
    q_ask0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    gamma_cross=1.0,
    rng=None,
    t_max=10_000.0,
    stop_at_zero=True,
    lambda_floor=lambda_floor,
    max_events=2_000_000,
):
    """Simulate bid and ask queues with coupled Hawkes removal intensities."""
    if rng is None:
        rng = np.random.default_rng()

    t = 0.0
    q_bid = int(q_bid0)
    q_ask = int(q_ask0)
    memory_bid = 0.0
    memory_ask = 0.0

    times = [t]
    bid_path = [q_bid]
    ask_path = [q_ask]
    bid_minus_path = []
    ask_minus_path = []
    event_types = []
    bid_removal_times = []
    ask_removal_times = []

    _, lbm, _, lam = two_queue_intensities(
        memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross, lambda_floor
    )
    bid_minus_path.append(lbm)
    ask_minus_path.append(lam)

    for _ in range(max_events):
        if stop_at_zero and (q_bid <= 0 or q_ask <= 0):
            break

        upper_rate = two_queue_total_rate_upper_bound(
            memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross, lambda_floor
        )
        dt = rng.exponential(1.0 / upper_rate)

        if t + dt > t_max:
            decay = math.exp(-beta * (t_max - t))
            memory_bid *= decay
            memory_ask *= decay
            t = t_max
            times.append(t)
            bid_path.append(q_bid)
            ask_path.append(q_ask)
            _, lbm, _, lam = two_queue_intensities(
                memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross, lambda_floor
            )
            bid_minus_path.append(lbm)
            ask_minus_path.append(lam)
            break

        decay = math.exp(-beta * dt)
        candidate_memory_bid = memory_bid * decay
        candidate_memory_ask = memory_ask * decay

        intensities = two_queue_intensities(
            candidate_memory_bid,
            candidate_memory_ask,
            mu_plus,
            mu_minus,
            gamma_cross,
            lambda_floor,
        )
        lambda_bid_plus, lambda_bid_minus, lambda_ask_plus, lambda_ask_minus = intensities
        candidate_total_rate = sum(intensities)

        if rng.random() > candidate_total_rate / upper_rate:
            t += dt
            memory_bid = candidate_memory_bid
            memory_ask = candidate_memory_ask
            continue

        t += dt
        memory_bid = candidate_memory_bid
        memory_ask = candidate_memory_ask

        u = rng.random() * candidate_total_rate
        if u < lambda_bid_plus:
            q_bid += 1
            memory_bid += alpha
            event_types.append("bid_arrival")
        elif u < lambda_bid_plus + lambda_bid_minus:
            q_bid -= 1
            memory_bid -= alpha
            event_types.append("bid_removal")
            bid_removal_times.append(t)
        elif u < lambda_bid_plus + lambda_bid_minus + lambda_ask_plus:
            q_ask += 1
            memory_ask += alpha
            event_types.append("ask_arrival")
        else:
            q_ask -= 1
            memory_ask -= alpha
            event_types.append("ask_removal")
            ask_removal_times.append(t)

        times.append(t)
        bid_path.append(q_bid)
        ask_path.append(q_ask)
        _, lbm, _, lam = two_queue_intensities(
            memory_bid, memory_ask, mu_plus, mu_minus, gamma_cross, lambda_floor
        )
        bid_minus_path.append(lbm)
        ask_minus_path.append(lam)

        if stop_at_zero and (q_bid <= 0 or q_ask <= 0):
            break
    else:
        raise RuntimeError("max_events reached before the simulation finished")

    if q_bid <= 0 and q_ask <= 0:
        first_queue = "both"
    elif q_bid <= 0:
        first_queue = "bid"
    elif q_ask <= 0:
        first_queue = "ask"
    else:
        first_queue = None

    return {
        "times": np.asarray(times),
        "bid": np.asarray(bid_path, dtype=int),
        "ask": np.asarray(ask_path, dtype=int),
        "lambda_bid_minus": np.asarray(bid_minus_path),
        "lambda_ask_minus": np.asarray(ask_minus_path),
        "event_types": np.asarray(event_types),
        "bid_removal_times": np.asarray(bid_removal_times),
        "ask_removal_times": np.asarray(ask_removal_times),
        "tau_first": t if first_queue is not None else np.nan,
        "first_queue": first_queue,
        "hit_zero": first_queue is not None,
    }

def estimate_coupled_stationary_removal_rates(
    n_paths,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    gamma_cross=1.0,
    rng=None,
    t_max=1_000.0,
    burn_in=200.0,
):
    if rng is None:
        rng = np.random.default_rng()

    estimates = np.empty((n_paths, 2))
    for k in range(n_paths):
        out = simulate_two_coupled_hawkes_queues(
            q_bid0=0,
            q_ask0=0,
            mu_plus=mu_plus,
            mu_minus=mu_minus,
            alpha=alpha,
            beta=beta,
            gamma_cross=gamma_cross,
            rng=rng,
            t_max=t_max,
            stop_at_zero=False,
            lambda_floor=lambda_floor,
        )
        estimates[k, 0] = np.sum(out["bid_removal_times"] >= burn_in) / (t_max - burn_in)
        estimates[k, 1] = np.sum(out["ask_removal_times"] >= burn_in) / (t_max - burn_in)

    return estimates


def sample_two_coupled_hawkes_first_hitting_times(
    n_samples,
    q0,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    gamma_cross=1.0,
    rng=None,
    t_max=10_000.0,
):
    if rng is None:
        rng = np.random.default_rng()

    tau = np.empty(n_samples)
    first_queue = np.empty(n_samples, dtype=object)
    hit = np.empty(n_samples, dtype=bool)

    for k in range(n_samples):
        out = simulate_two_coupled_hawkes_queues(
            q_bid0=q0,
            q_ask0=q0,
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
        tau[k] = out["tau_first"]
        first_queue[k] = out["first_queue"]
        hit[k] = out["hit_zero"]

    return tau, first_queue, hit


def simulate_poisson_one_queue_until_zero(q0, lambda_plus, lambda_minus, rng=None, max_events=1_000_000):
    if rng is None:
        rng = np.random.default_rng()
    if q0 <= 0:
        return 0.0

    rate = lambda_plus + lambda_minus
    p_up = lambda_plus / rate
    t = 0.0
    q = int(q0)

    for _ in range(max_events):
        t += rng.exponential(1.0 / rate)
        q += 1 if rng.random() < p_up else -1
        if q <= 0:
            return t

    raise RuntimeError("max_events reached before hitting zero")


def sample_poisson_hitting_times(n_samples, q0, lambda_plus, lambda_minus, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.array([
        simulate_poisson_one_queue_until_zero(q0, lambda_plus, lambda_minus, rng=rng)
        for _ in range(n_samples)
    ])