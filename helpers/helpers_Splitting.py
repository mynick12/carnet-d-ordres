from helpers.helpers_QRNIID import *
from helpers.helpers_QR2LIM import *

def initial_particle(q0):
    return {
        "t": 0.0,
        "q_bid": int(q0),   # correspond à N^{+1}
        "q_ask": int(q0),   # correspond à N^{-1}
        "memory_bid": 0.0,
        "memory_ask": 0.0,
        "bid_removal_times": [],
        "ask_removal_times": [],
        "alive": True,
    }


def simulate_until_bid_level(
    particle,
    target_level,
    mu_plus,
    mu_minus,
    alpha,
    beta,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    max_events=2_000_000,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Copier l'état pour ne pas modifier l'original
    p = {
        "t": particle["t"],
        "q_bid": particle["q_bid"],
        "q_ask": particle["q_ask"],
        "memory_bid": particle["memory_bid"],
        "memory_ask": particle["memory_ask"],
        "bid_removal_times": list(particle["bid_removal_times"]),
        "ask_removal_times": list(particle["ask_removal_times"]),
        "alive": particle["alive"],
    }

    if not p["alive"]:
        return p, False

    for _ in range(max_events):

        # Succès : on a atteint le niveau demandé
        if p["q_bid"] <= target_level:
            return p, True

        # Échec : l'autre côté s'est vidé avant
        if p["q_ask"] <= 0:
            p["alive"] = False
            return p, False

        if p["t"] >= t_max:
            p["alive"] = False
            return p, False

        upper_rate = two_queue_total_rate_upper_bound(
            p["memory_bid"],
            p["memory_ask"],
            mu_plus,
            mu_minus,
            gamma_cross,
            lambda_floor,
        )

        dt = rng.exponential(1.0 / upper_rate)

        if p["t"] + dt > t_max:
            decay = math.exp(-beta * (t_max - p["t"]))
            p["memory_bid"] *= decay
            p["memory_ask"] *= decay
            p["t"] = t_max
            p["alive"] = False
            return p, False

        decay = math.exp(-beta * dt)
        candidate_memory_bid = p["memory_bid"] * decay
        candidate_memory_ask = p["memory_ask"] * decay

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

        # Rejet par amincissement
        if rng.random() > candidate_total_rate / upper_rate:
            p["t"] += dt
            p["memory_bid"] = candidate_memory_bid
            p["memory_ask"] = candidate_memory_ask
            continue

        # Acceptation de l'événement
        p["t"] += dt
        p["memory_bid"] = candidate_memory_bid
        p["memory_ask"] = candidate_memory_ask

        u = rng.random() * candidate_total_rate

        if u < lambda_bid_plus:
            # ajout sur N^{+1}
            p["q_bid"] += 1
            p["memory_bid"] += alpha

        elif u < lambda_bid_plus + lambda_bid_minus:
            # retrait sur N^{+1}
            p["q_bid"] -= 1
            p["memory_bid"] -= alpha
            p["bid_removal_times"].append(p["t"])

        elif u < lambda_bid_plus + lambda_bid_minus + lambda_ask_plus:
            # ajout sur N^{-1}
            p["q_ask"] += 1
            p["memory_ask"] += alpha

        else:
            # retrait sur N^{-1}
            p["q_ask"] -= 1
            p["memory_ask"] -= alpha
            p["ask_removal_times"].append(p["t"])

    raise RuntimeError("max_events reached")


def splitting_estimator(
    h,
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
    a_second,
    b_second,
    gamma_cross=1.0,
    lambda_floor=1e-8,
    t_max=10_000.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    particles = [initial_particle(q0) for _ in range(n_particles)]

    survival_fractions = []

    for level in levels:
        new_particles = []
        successes = []

        for p in particles:
            p_new, success = simulate_until_bid_level(
                p,
                target_level=level,
                mu_plus=mu_plus,
                mu_minus=mu_minus,
                alpha=alpha,
                beta=beta,
                gamma_cross=gamma_cross,
                lambda_floor=lambda_floor,
                t_max=t_max,
                rng=rng,
            )

            if success:
                successes.append(p_new)

        frac = len(successes) / n_particles
        survival_fractions.append(frac)

        print(f"Niveau {level}: {len(successes)} / {n_particles} succès, fraction = {frac:.4f}")

        if len(successes) == 0:
            return {
                "p_hit_plus1_zero": 0.0,
                "p_cond_second_low": np.nan,
                "p_joint": 0.0,
                "survival_fractions": survival_fractions,
                "final_particles": [],
                "plus2_values": np.array([]),
            }

        # Resampling : on clone les trajectoires survivantes
        indices = rng.integers(0, len(successes), size=n_particles)
        particles = [successes[i].copy() for i in indices]

        # Attention : copy() ne copie pas profondément les listes
        for p in particles:
            p["bid_removal_times"] = list(p["bid_removal_times"])
            p["ask_removal_times"] = list(p["ask_removal_times"])

    # À ce stade, les particules ont atteint N^{+1}=0 avant N^{-1}=0
    final_particles = particles

    plus2_values = np.empty(n_particles)

    for i, p in enumerate(final_particles):
        plus2_values[i] = simulate_reactive_second_limit_until_time(
            q2_0=q2_0,
            horizon=p["t"],
            driver_removal_times=p["bid_removal_times"],
            mu_second_add=lambda_second_add,
            lambda_second_remove=lambda_second_remove,
            a=a_second,
            b=b_second,
            rng=rng,
        )

    indicators = plus2_values <= h

    p_hit_plus1_zero = np.prod(survival_fractions)
    p_cond_second_low = np.mean(indicators)
    p_joint = p_hit_plus1_zero * p_cond_second_low

    return {
        "p_hit_plus1_zero": p_hit_plus1_zero,
        "p_cond_second_low": p_cond_second_low,
        "p_joint": p_joint,
        "survival_fractions": survival_fractions,
        "final_particles": final_particles,
        "plus2_values": plus2_values,
    }







