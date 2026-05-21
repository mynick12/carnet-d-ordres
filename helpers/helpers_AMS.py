import math
import numpy as np

lambda_floor = 1e-8

def _two_queue_intensities(memory_plus, memory_minus, mu_plus, mu_minus, gamma_cross=1.0, lambda_floor=lambda_floor):
    plus_minus_signal = memory_plus + gamma_cross * memory_minus
    minus_minus_signal = memory_minus + gamma_cross * memory_plus
    return (
        mu_plus,
        max(mu_minus + plus_minus_signal, lambda_floor),
        mu_plus,
        max(mu_minus + minus_minus_signal, lambda_floor),
    )


def _two_queue_upper(memory_plus, memory_minus, mu_plus, mu_minus, gamma_cross=1.0, lambda_floor=lambda_floor):
    plus_signal = memory_plus + gamma_cross * memory_minus
    minus_signal = memory_minus + gamma_cross * memory_plus
    upper_plus_minus = max(mu_minus + max(plus_signal, 0.0), mu_minus, lambda_floor)
    upper_minus_minus = max(mu_minus + max(minus_signal, 0.0), mu_minus, lambda_floor)
    return 2.0 * mu_plus + upper_plus_minus + upper_minus_minus


def simulate_reactive_path_stateful(
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
    initial_state=None,
    history_prefix=None,
    max_events=2_000_000,
):
    if rng is None:
        rng = np.random.default_rng()

    if initial_state is None:
        t = 0.0
        q_plus1 = int(q1_0)
        q_minus1 = int(q1_0)
        q_plus2 = int(q2_0)
        memory_plus = 0.0
        memory_minus = 0.0
        z_plus2 = 0.0
        history = []
    else:
        t = float(initial_state["t"])
        q_plus1 = int(initial_state["q_plus1"])
        q_minus1 = int(initial_state["q_minus1"])
        q_plus2 = int(initial_state["q_plus2"])
        memory_plus = float(initial_state["memory_plus"])
        memory_minus = float(initial_state["memory_minus"])
        z_plus2 = float(initial_state["z_plus2"])
        history = list(history_prefix) if history_prefix is not None else []

    def current_state(event):
        return {
            "t": t,
            "q_plus1": q_plus1,
            "q_minus1": q_minus1,
            "q_plus2": q_plus2,
            "memory_plus": memory_plus,
            "memory_minus": memory_minus,
            "z_plus2": z_plus2,
            "event": event,
        }

    if not history:
        history.append(current_state("start"))

    for _ in range(max_events):
        if q_plus1 <= 0:
            first_empty = "plus1"
            break
        if q_minus1 <= 0:
            first_empty = "minus1"
            break

        upper_first = _two_queue_upper(memory_plus, memory_minus, mu_plus, mu_minus, gamma_cross, lambda_floor)
        active_second_remove = lambda_second_remove if q_plus2 > 0 else 0.0
        upper_second = lambda_second_add + max(z_plus2, 0.0) + active_second_remove
        upper_rate = upper_first + upper_second
        if upper_rate <= 0:
            first_empty = None
            break

        dt = rng.exponential(1.0 / upper_rate)
        if t + dt > t_max:
            decay_first = math.exp(-beta * (t_max - t))
            decay_second = math.exp(-b_second * (t_max - t))
            memory_plus *= decay_first
            memory_minus *= decay_first
            z_plus2 *= decay_second
            t = t_max
            history.append(current_state("t_max"))
            first_empty = None
            break

        decay_first = math.exp(-beta * dt)
        decay_second = math.exp(-b_second * dt)
        cand_memory_plus = memory_plus * decay_first
        cand_memory_minus = memory_minus * decay_first
        cand_z_plus2 = z_plus2 * decay_second

        l_p1_plus, l_p1_minus, l_m1_plus, l_m1_minus = _two_queue_intensities(
            cand_memory_plus, cand_memory_minus, mu_plus, mu_minus, gamma_cross, lambda_floor
        )
        if q_plus1 <= 0:
            l_p1_minus = 0.0
        if q_minus1 <= 0:
            l_m1_minus = 0.0

        l_p2_plus = lambda_second_add + cand_z_plus2
        l_p2_minus = lambda_second_remove if q_plus2 > 0 else 0.0
        true_rates = np.array([l_p1_plus, l_p1_minus, l_m1_plus, l_m1_minus, l_p2_plus, l_p2_minus], dtype=float)
        true_total = true_rates.sum()

        # Thinning rejection.
        if rng.random() > true_total / upper_rate:
            t += dt
            memory_plus = cand_memory_plus
            memory_minus = cand_memory_minus
            z_plus2 = cand_z_plus2
            continue

        t += dt
        memory_plus = cand_memory_plus
        memory_minus = cand_memory_minus
        z_plus2 = cand_z_plus2

        u = rng.random() * true_total
        idx = int(np.searchsorted(np.cumsum(true_rates), u, side="right"))

        if idx == 0:
            q_plus1 += 1
            memory_plus += alpha
            event = "plus1_add"
        elif idx == 1:
            q_plus1 -= 1
            memory_plus -= alpha
            z_plus2 += a_second
            event = "plus1_remove"
        elif idx == 2:
            q_minus1 += 1
            memory_minus += alpha
            event = "minus1_add"
        elif idx == 3:
            q_minus1 -= 1
            memory_minus -= alpha
            event = "minus1_remove"
        elif idx == 4:
            q_plus2 += 1
            event = "plus2_add"
        else:
            q_plus2 -= 1
            event = "plus2_remove"

        history.append(current_state(event))
    else:
        raise RuntimeError("max_events reached before the simulation finished")

    q2_path = np.array([s["q_plus2"] for s in history], dtype=int)
    min_q2 = int(np.min(q2_path))
    score = int(q2_0 - min_q2)

    return {
        "history": history,
        "tau": t if first_empty is not None else np.nan,
        "hit_zero": first_empty is not None,
        "first_empty": first_empty,
        "plus2_at_tau": q_plus2 if first_empty is not None else np.nan,
        "min_plus2": min_q2,
        "score": score,
    }


def first_crossing_prefix(path, level):
    history = path["history"]
    q2_initial = history[0]["q_plus2"]
    best = -np.inf
    for k, state in enumerate(history):
        best = max(best, q2_initial - state["q_plus2"])
        if best >= level:
            prefix = history[: k + 1]
            st = prefix[-1]
            init_state = {key: st[key] for key in ["t", "q_plus1", "q_minus1", "q_plus2", "memory_plus", "memory_minus", "z_plus2"]}
            return init_state, prefix
    # fallback: return full path end
    st = history[-1]
    init_state = {key: st[key] for key in ["t", "q_plus1", "q_minus1", "q_plus2", "memory_plus", "memory_minus", "z_plus2"]}
    return init_state, history


def ams_low_second_limit(
    h,
    target_first_empty,
    n_particles,
    kill_fraction,
    max_iterations,
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
    if rng is None:
        rng = np.random.default_rng()
    if target_first_empty not in {"plus1", "minus1"}:
        raise ValueError("target_first_empty must be 'plus1' or 'minus1'")

    final_level = int(q2_0 - h)
    if final_level <= 0:
        raise ValueError("h must be smaller than q2_0 for this score q2_0 - min(N+2)")

    particles = [
        simulate_reactive_path_stateful(
            q1_0=q1_0, q2_0=q2_0,
            mu_plus=mu_plus, mu_minus=mu_minus, alpha=alpha, beta=beta,
            lambda_second_add=lambda_second_add, lambda_second_remove=lambda_second_remove,
            a_second=a_second, b_second=b_second, gamma_cross=gamma_cross,
            rng=rng, t_max=t_max,
        )
        for _ in range(n_particles)
    ]

    k_kill = max(1, int(np.floor(kill_fraction * n_particles)))
    log_weight = 0.0
    levels = []

    for it in range(max_iterations):
        scores = np.array([p["score"] for p in particles], dtype=float)
        scores_sorted = np.sort(scores)
        level = scores_sorted[k_kill - 1]
        level = int(level)
        if level >= final_level:
            break

        # Kill exactly the k_kill particles with the lowest scores.
        # This avoids non-monotone adaptive levels caused by random tie handling.
        killed_idx = np.argsort(scores, kind="mergesort")[:k_kill]
        killed_set = set(map(int, killed_idx))
        survivor_idx = np.array([i for i in range(n_particles) if i not in killed_set])
        log_weight += math.log((n_particles - len(killed_idx)) / n_particles)
        levels.append(level)

        for idx in killed_idx:
            parent_idx = int(rng.choice(survivor_idx))
            parent = particles[parent_idx]
            init_state, prefix = first_crossing_prefix(parent, level)
            particles[idx] = simulate_reactive_path_stateful(
                q1_0=q1_0, q2_0=q2_0,
                mu_plus=mu_plus, mu_minus=mu_minus, alpha=alpha, beta=beta,
                lambda_second_add=lambda_second_add, lambda_second_remove=lambda_second_remove,
                a_second=a_second, b_second=b_second, gamma_cross=gamma_cross,
                rng=rng, t_max=t_max,
                initial_state=init_state,
                history_prefix=prefix,
            )

    # Final correction: only count paths satisfying the actual terminal event and the conditioning side.
    indicators = np.array([
        (p["hit_zero"] and p["first_empty"] == target_first_empty and p["plus2_at_tau"] <= h)
        for p in particles
    ], dtype=float)
    p_joint = math.exp(log_weight) * indicators.mean()

    # Denominator is not rare; estimate it with the same final particle population as a rough value.
    # For final reporting, prefer a separate naive estimate with many particles.
    denom_rough = np.mean([p["hit_zero"] and p["first_empty"] == target_first_empty for p in particles])

    return {
        "p_joint_estimate": p_joint,
        "denom_rough": denom_rough,
        "conditional_rough": p_joint / denom_rough if denom_rough > 0 else np.nan,
        "levels": levels,
        "final_level": final_level,
        "particles": particles,
        "final_fraction": indicators.mean(),
        "log_weight": log_weight,
    }


def estimate_denominator_naive(n_samples, target_first_empty, sim_kwargs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    count = 0
    hit = 0
    for _ in range(n_samples):
        p = simulate_reactive_path_stateful(rng=rng, **sim_kwargs)
        hit += int(p["hit_zero"])
        count += int(p["hit_zero"] and p["first_empty"] == target_first_empty)
    return count / n_samples, hit / n_samples


def naive_conditional_probability(h, target_first_empty, n_samples, sim_kwargs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    numerator = 0
    denominator = 0
    values = []

    for _ in range(n_samples):
        path = simulate_reactive_path_stateful(rng=rng, **sim_kwargs)
        if path["hit_zero"] and path["first_empty"] == target_first_empty:
            denominator += 1
            values.append(path["plus2_at_tau"])
            if path["plus2_at_tau"] <= h:
                numerator += 1

    p_hat = numerator / denominator if denominator > 0 else np.nan
    se_hat = np.sqrt(p_hat * (1.0 - p_hat) / denominator) if denominator > 0 else np.nan

    return {
        "h": h,
        "target_first_empty": target_first_empty,
        "n_samples": n_samples,
        "numerator": numerator,
        "denominator": denominator,
        "p_hat": p_hat,
        "se_hat": se_hat,
        "values": np.array(values),
    }


def repeat_ams(h, target_first_empty, n_repeats, n_particles, kill_fraction, max_iterations, denom, sim_kwargs, seed=2026):
    estimates = []
    levels_list = []

    for r in range(n_repeats):
        local_rng = np.random.default_rng(seed + r)
        out = ams_low_second_limit(
            h=h,
            target_first_empty=target_first_empty,
            n_particles=n_particles,
            kill_fraction=kill_fraction,
            max_iterations=max_iterations,
            rng=local_rng,
            **sim_kwargs,
        )
        estimates.append(out["p_joint_estimate"] / denom if denom > 0 else np.nan)
        levels_list.append(out["levels"])

    estimates = np.array(estimates, dtype=float)
    return {
        "estimates": estimates,
        "mean": np.nanmean(estimates),
        "std": np.nanstd(estimates, ddof=1),
        "se": np.nanstd(estimates, ddof=1) / np.sqrt(np.sum(~np.isnan(estimates))),
        "levels": levels_list,
    }

