"""
Evolution Strategy (mu, lambda) with self-adaptive step sizes and recombination
Benchmark function: 10-dimensional Rastrigin

Reference: (Schwefel 1977)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

N           = 10        # problem dimensionality
MU          = 50        # number of parents
LAM         = 400       # number of offspring per generation
SIGMA0      = 1.0       # initial step size
SIGMA_MIN   = 1e-5      # minimum step size (epsilon_0)
X_MIN       = -5.12     # domain lower bound
X_MAX       =  5.12     # domain upper bound
MAX_EVALS   = 50_000    # maximum number of J evaluations
EPSILON     = 0.05      # success criterion: J <= EPSILON  <->  F >= -EPSILON
NC_THRESHOLD = 0.20     # convexity threshold: nc < NC_THRESHOLD -> convex convergence
N_RUNS      = 30        # number of independent runs
BASE_SEED   = 42        # first seed (runs use 42, 43, ..., 71)

# step-size mutation factors (Schwefel / GA_Aula4 p.26)
G1 = 1.0 / np.sqrt(2 * N)          # global perturbation
G2 = 1.0 / np.sqrt(2 * np.sqrt(N)) # local perturbation


# 1. RASTRIGIN FUNCTION
def rastrigin(x):
    A = 10.0
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def fitness(x):
    """
    Fitness function F(x) = -J(x).
    The ES maximises F(x); the minimum of J corresponds to the maximum of F.
    """
    return -rastrigin(x)


# 2. POPULATION INITIALISATION
def initialise_population(rng):
    population = []
    for _ in range(MU):
        individual = {
            'x':       rng.uniform(X_MIN, X_MAX, N),
            'sigma':   np.full(N, SIGMA0),
            'fitness': -np.inf,   # fitness: higher is better
        }
        population.append(individual)
    return population


# 3. RECOMBINATION
def recombine(parent1, parent2, rng):
    """
    x     -> discrete recombination (coin-flip per dimension)
    sigma -> global intermediate recombination (mean of both parents)
    """
    mask = rng.random(N) < 0.5
    offspring = {
        'x':       np.where(mask, parent1['x'], parent2['x']),
        'sigma':   0.5 * (parent1['sigma'] + parent2['sigma']),
        'fitness': -np.inf,
    }
    return offspring


# 4. MUTATION
def mutate(individual, rng):
    """
    sigma_i' = sigma_i * exp( G1*N(0,1) + G2*N_i(0,1) )
    x_i'     = x_i + sigma_i' * N_i(0,1)
    if sigma_i' < SIGMA_MIN  ->  sigma_i' = SIGMA_MIN
    """
    z_global = rng.standard_normal()
    z_local  = rng.standard_normal(N)

    new_sigma = individual['sigma'] * np.exp(G1 * z_global + G2 * z_local)
    new_sigma = np.maximum(new_sigma, SIGMA_MIN)

    z_x    = rng.standard_normal(N)
    new_x  = individual['x'] + new_sigma * z_x
    new_x  = np.clip(new_x, X_MIN, X_MAX)

    return {
        'x':       new_x,
        'sigma':   new_sigma,
        'fitness': -np.inf,
    }


# 5. MAIN ES ALGORITHM
def run_es(seed):
    """
    One independent run of the (mu, lambda) ES.
    Returns a dictionary with results and history.
    """
    rng      = np.random.default_rng(seed)
    n_evals  = 0
    history_best  = []
    history_sigma = []
    history_evals = []

    # initialise and evaluate population
    population = initialise_population(rng)
    for ind in population:
        ind['fitness'] = fitness(ind['x'])
        n_evals += 1

    best = max(population, key=lambda i: i['fitness'])
    best = {'x': best['x'].copy(), 'sigma': best['sigma'].copy(),
            'fitness': best['fitness']}

    history_best.append(best['fitness'])
    history_sigma.append(float(np.mean(population[0]['sigma'])))
    history_evals.append(n_evals)

    # evolutionary loop
    while n_evals < MAX_EVALS:
        offspring_list = []

        for _ in range(LAM):
            idx1, idx2 = rng.choice(MU, size=2, replace=False)
            child = recombine(population[idx1], population[idx2], rng)
            child = mutate(child, rng)
            child['fitness'] = fitness(child['x'])
            n_evals += 1
            offspring_list.append(child)

            if child['fitness'] >= -EPSILON:   # F >= -eps  <->  J <= eps
                if child['fitness'] > best['fitness']:
                    best = {'x': child['x'].copy(), 'sigma': child['sigma'].copy(),
                            'fitness': child['fitness']}
                history_best.append(best['fitness'])
                history_sigma.append(float(np.mean(
                    [np.mean(f['sigma']) for f in offspring_list])))
                history_evals.append(n_evals)
                return {
                    'best_x': best['x'], 'best_fitness': best['fitness'],
                    'n_evals': n_evals, 'success': True,
                    'history_best':  history_best,
                    'history_sigma': history_sigma,
                    'history_evals': history_evals,
                    'final_population': population,
                }

            if n_evals >= MAX_EVALS:
                break

        # (mu, lambda) selection: maximisation -> descending sort, keep top MU
        offspring_list.sort(key=lambda i: i['fitness'], reverse=True)
        population = offspring_list[:MU]

        if population[0]['fitness'] > best['fitness']:
            best = {'x': population[0]['x'].copy(),
                    'sigma': population[0]['sigma'].copy(),
                    'fitness': population[0]['fitness']}

        history_best.append(best['fitness'])
        history_sigma.append(float(np.mean(
            [np.mean(ind['sigma']) for ind in population])))
        history_evals.append(n_evals)

    return {
        'best_x': best['x'], 'best_fitness': best['fitness'],
        'n_evals': n_evals, 'success': best['fitness'] >= -EPSILON,
        'history_best':  history_best,
        'history_sigma': history_sigma,
        'history_evals': history_evals,
        'final_population': population,
    }


# 6. CONVEXITY ANALYSIS
def analyse_convexity(history_evals, history_fitness, n_points=200):
    """
    Analyses the convexity of a run's convergence curve via the discrete
    second derivative.

    A perfectly convex curve (smooth descent) has d²F/dt² >= 0 throughout.
    Plateaus followed by drops produce regions with d²F/dt² < 0, indicating
    the algorithm was trapped and later escaped.

    Returns:
      non_convex_fraction -- proportion of the trajectory where d2 < 0
                             (0 = fully convex, 1 = fully non-convex)
      intensity           -- sum of negative d2 values (magnitude of inflections)
    """
    if len(history_evals) < 3:
        return np.nan, np.nan

    grid = np.linspace(history_evals[0], history_evals[-1], n_points)
    f    = np.interp(grid, history_evals, history_fitness)
    d2   = np.diff(f, n=2)

    non_convex_fraction = float(np.mean(d2 < 0))
    intensity           = float(np.sum(np.minimum(d2, 0)))
    return non_convex_fraction, intensity


# 7. EXPERIMENTAL PROTOCOL — 30 RUNS
def run_experiment(label="ES"):
    seeds   = list(range(BASE_SEED, BASE_SEED + N_RUNS))
    results = []

    print(f"\n{'='*50}")
    print(f"{label} (mu={MU}, lambda={LAM}) — Rastrigin d={N}")
    print(f"Budget: {MAX_EVALS:,} evaluations  |  eps = {EPSILON}")
    print(f"{'='*50}")

    for i, seed in enumerate(seeds):
        res = run_es(seed)

        # per-run convexity analysis
        nc_frac, intens = analyse_convexity(
            res['history_evals'], res['history_best'])
        res['conv_fraction']  = nc_frac
        res['conv_intensity'] = intens

        results.append(res)
        cost   = -res['best_fitness']
        status = "ok" if res['success'] else f"J={cost:.4f}"
        print(f"  Run {i+1:02d} | evals={res['n_evals']:>7,} | {status}"
              f" | nc={nc_frac:.2f}  int={intens:.4f}")

    # MBF and std computed over cost J = -F
    cost_runs    = [-r['best_fitness'] for r in results]
    runs_ok      = [r for r in results if r['success']]
    runs_fail    = [r for r in results if not r['success']]
    evals_ok     = [r['n_evals'] for r in runs_ok]

    sr      = len(runs_ok) / N_RUNS
    mbf     = np.mean(cost_runs)
    mbf_std = np.std(cost_runs)
    aes     = np.mean(evals_ok) if runs_ok else None
    aes_std = np.std(evals_ok)  if runs_ok else None

    # aggregated convexity: mean by group (success / failure / total)
    nc_total = [r['conv_fraction']  for r in results  if not np.isnan(r['conv_fraction'])]
    nc_ok    = [r['conv_fraction']  for r in runs_ok  if not np.isnan(r['conv_fraction'])]
    nc_fail  = [r['conv_fraction']  for r in runs_fail if not np.isnan(r['conv_fraction'])]
    int_total = [r['conv_intensity'] for r in results  if not np.isnan(r['conv_intensity'])]

    conv_mean   = np.mean(nc_total)  if nc_total  else np.nan
    conv_ok     = np.mean(nc_ok)     if nc_ok     else np.nan
    conv_fail_m = np.mean(nc_fail)   if nc_fail   else np.nan
    intens_mean = np.mean(int_total) if int_total else np.nan

    # extended SR: formal success OR failure with convex curve (nc < NC_THRESHOLD)
    runs_conv = [r for r in runs_fail
                 if not np.isnan(r['conv_fraction'])
                 and r['conv_fraction'] < NC_THRESHOLD]
    n_extended_ok = len(runs_ok) + len(runs_conv)
    sr_conv       = n_extended_ok / N_RUNS

    print(f"\n  SR       = {sr*100:.1f}%  ({len(runs_ok)}/{N_RUNS})  [J <= eps]")
    print(f"  SR_conv  = {sr_conv*100:.1f}%  ({n_extended_ok}/{N_RUNS})"
          f"  [J <= eps  OR  nc < {NC_THRESHOLD}]")
    print(f"  MBF = {mbf:.4f} +/- {mbf_std:.4f}")
    print(f"  AES = {f'{aes:.0f} +/- {aes_std:.0f}' if aes else 'N/A'}")
    print(f"\n  Convexity (non-convex fraction):")
    print(f"    Total   = {conv_mean:.3f}")
    print(f"    Success = {conv_ok:.3f}")
    print(f"    Failure = {conv_fail_m:.3f}")
    print(f"    Mean intensity = {intens_mean:.4f}")

    return {
        'label': label, 'results': results, 'seeds': seeds,
        'sr': sr, 'sr_conv': sr_conv, 'n_conv': len(runs_conv),
        'mbf': mbf, 'mbf_std': mbf_std,
        'aes': aes, 'aes_std': aes_std, 'n_success': len(runs_ok),
        'conv_mean': conv_mean, 'conv_ok': conv_ok, 'conv_fail': conv_fail_m,
        'intens_mean': intens_mean,
    }


# 8. VISUALISATIONS
def plot_convergence(exp, ax, color):
    grid   = np.linspace(0, MAX_EVALS, 500)
    curves = []
    for res in exp['results']:
        if len(res['history_evals']) < 2:
            continue
        # history_best stores F = -J; plot F directly (rises toward 0)
        curves.append(np.interp(grid, res['history_evals'], res['history_best']))
    curves = np.array(curves)
    mean   = np.mean(curves, axis=0)
    ax.plot(grid, mean, color=color, label=exp['label'], linewidth=2)


def plot_sigma(exp, ax, color):
    results = exp['results']
    max_len = max(len(r['history_sigma']) for r in results)
    curves  = []
    for r in results:
        arr = np.array(r['history_sigma'])
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=arr[-1])
        curves.append(arr)
    curves = np.array(curves)
    mean   = np.mean(curves, axis=0)
    gens   = np.arange(max_len)
    ax.plot(gens, mean, color=color, label=exp['label'], linewidth=2)


if __name__ == "__main__":

    exp_es = run_experiment(label="ES")

    # summary table
    print("\n" + "="*50)
    print("SUMMARY TABLE — ES (mu, lambda)  |  Rastrigin d=10")
    print("="*50)
    df = pd.DataFrame([{
        "SR (%)":  f"{exp_es['sr']*100:.1f}",
        "MBF":     f"{exp_es['mbf']:.4f}",
        "Std MBF": f"{exp_es['mbf_std']:.4f}",
        "AES":     f"{exp_es['aes']:.0f}" if exp_es['aes'] else "N/A",
        "Std AES": f"{exp_es['aes_std']:.0f}" if exp_es['aes_std'] else "N/A",
    }])
    print(df.to_string(index=False))

    # figure 1: convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_convergence(exp_es, ax, color="#5f7fbf")
    ax.set_xlabel("J(x) evaluations")
    ax.set_ylabel("Best F(x) = -J(x)")
    ax.set_title("Convergence — ES  |  Rastrigin d=10")
    ax.axhline(y=-EPSILON, color="gray", linestyle="--",
               linewidth=1, label=f"-eps = {-EPSILON}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("es_convergence.png", dpi=150, bbox_inches="tight")

    # figure 2: sigma trajectory
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_sigma(exp_es, ax2, color="#5f7fbf")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean sigma")
    ax2.set_title("Step-size trajectory — ES  d=10")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("es_sigma.png", dpi=150, bbox_inches="tight")

    print("\nFigures saved: es_convergence.png | es_sigma.png")
