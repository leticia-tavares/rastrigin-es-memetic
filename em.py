"""
Memetic Algorithm = (mu, lambda) ES + greedy local search
Benchmark function: 10-dimensional Rastrigin

Reference:  (Moscato 1989)
  - Evolutionary cycle (ES) + learning step (greedy local search)
  - Lamarckian effect: improved position is carried into the next generation
  Eiben & Smith (2015), Ch. 10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EXPERIMENT PARAMETERS
# (ES identical to es.py — same seeds, same budget)

N           = 10        # problem dimensionality
MU          = 50        # number of parents
LAM         = 400       # number of offspring per generation
SIGMA0      = 1.0       # initial step size
SIGMA_MIN   = 1e-5      # minimum step size
X_MIN       = -5.12
X_MAX       =  5.12
MAX_EVALS   = 50_000    # maximum budget — shared between ES and local search
EPSILON     = 0.05      # success criterion: J <= EPSILON  <->  F >= -EPSILON
NC_THRESHOLD = 0.20     # convexity threshold: nc < NC_THRESHOLD -> convex convergence
N_RUNS      = 30
BASE_SEED   = 42

G1 = 1.0 / np.sqrt(2 * N)
G2 = 1.0 / np.sqrt(2 * np.sqrt(N))

# local search parameters
# Budget per generation: LAM + K_LOCAL * DEPTH = 400 + 3*50 = 550 evals
# -> ~90 generations within the 50k total budget
K_LOCAL = 3    # top offspring that receive local search (~20% of MU)
DEPTH   = 50   # attempts per individual (search depth)
SIGMA_LOCAL = 0.5  # Gaussian perturbation step for local search
               # calibrated empirically: large enough to attempt crossing
               # the barriers between Rastrigin basins (~1.0 apart)


# 1. RASTRIGIN FUNCTION
def rastrigin(x):
    A = 10.0
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def fitness(x):
    """
    Fitness function F(x) = -J(x).
    The MA maximises F(x); the minimum of J corresponds to the maximum of F.
    """
    return -rastrigin(x)

# 2. INITIALISATION, RECOMBINATION, MUTATION
# (identical to ES)
def initialise_population(rng):
    population = []
    for _ in range(MU):
        population.append({
            'x':       rng.uniform(X_MIN, X_MAX, N),
            'sigma':   np.full(N, SIGMA0),
            'fitness': -np.inf,   # fitness: higher is better
        })
    return population


def recombine(parent1, parent2, rng):
    mask = rng.random(N) < 0.5
    return {
        'x':       np.where(mask, parent1['x'], parent2['x']),
        'sigma':   0.5 * (parent1['sigma'] + parent2['sigma']),
        'fitness': -np.inf,
    }


def mutate(individual, rng):
    z_global  = rng.standard_normal()
    z_local   = rng.standard_normal(N)
    new_sigma = individual['sigma'] * np.exp(G1 * z_global + G2 * z_local)
    new_sigma = np.maximum(new_sigma, SIGMA_MIN)
    new_x     = individual['x'] + new_sigma * rng.standard_normal(N)
    new_x     = np.clip(new_x, X_MIN, X_MAX)
    return {'x': new_x, 'sigma': new_sigma, 'fitness': -np.inf}


# 3. GREEDY LOCAL SEARCH
def local_search(individual, rng):
    """
    Greedy hill-climbing with Gaussian perturbation — operates on J(x).
    The local search minimises J directly (cost comparisons are more natural
    here). The returned individual has fitness = -J(x).

    Lamarckian effect: improved position is preserved for the next generation.
    The ES sigma is not modified by the local search.

    Returns: (improved individual, number of evaluations consumed)
    """
    x_current = individual['x'].copy()
    j_current = rastrigin(x_current)   # work with cost J internally
    n_evals   = 0

    for _ in range(DEPTH):
        x_candidate = x_current + SIGMA_LOCAL * rng.standard_normal(N)
        x_candidate = np.clip(x_candidate, X_MIN, X_MAX)
        j_candidate = rastrigin(x_candidate)
        n_evals += 1
        if j_candidate < j_current:    # greedy: accept improvement in J
            x_current = x_candidate
            j_current = j_candidate

    return {
        'x':       x_current,
        'sigma':   individual['sigma'].copy(),
        'fitness': -j_current,          # return fitness F = -J
    }, n_evals


# 4. MAIN MEMETIC ALGORITHM
def run_memetic(seed):
    """
    Memetic Algorithm = (mu, lambda) ES + greedy local search on K_LOCAL
    best offspring.

    Per-generation flow (GA_Aula8 Section 9.3):
      1. Generate lambda offspring via ES recombination + mutation
      2. Evaluate all lambda offspring
      3. Sort offspring by fitness
      4. Apply greedy local search to the K_LOCAL best
         (local search evaluations count against the total budget)
      5. Re-sort and select mu best -> next generation
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

    def record_and_return(success):
        history_best.append(best['fitness'])
        history_sigma.append(float(np.mean(
            [np.mean(f['sigma']) for f in offspring_list])))
        history_evals.append(n_evals)
        return {
            'best_x': best['x'], 'best_fitness': best['fitness'],
            'n_evals': n_evals, 'success': success,
            'history_best':  history_best,
            'history_sigma': history_sigma,
            'history_evals': history_evals,
            'final_population': population,
        }

    # evolutionary loop
    while n_evals < MAX_EVALS:
        offspring_list = []

        # -- ES step: generate lambda offspring --
        for _ in range(LAM):
            idx1, idx2 = rng.choice(MU, size=2, replace=False)
            child = recombine(population[idx1], population[idx2], rng)
            child = mutate(child, rng)
            child['fitness'] = fitness(child['x'])
            n_evals += 1
            offspring_list.append(child)
            if n_evals >= MAX_EVALS:
                break

        # sort descending to identify the best before local search
        offspring_list.sort(key=lambda i: i['fitness'], reverse=True)

        # -- local search step: refine the K_LOCAL best offspring --
        for j in range(min(K_LOCAL, len(offspring_list))):
            if n_evals >= MAX_EVALS:
                break

            offspring_list[j], ls_evals = local_search(offspring_list[j], rng)
            n_evals += ls_evals

            # update global best
            if offspring_list[j]['fitness'] > best['fitness']:
                best = {'x': offspring_list[j]['x'].copy(),
                        'sigma': offspring_list[j]['sigma'].copy(),
                        'fitness': offspring_list[j]['fitness']}

            # check success criterion: F >= -eps  <->  J <= eps
            if best['fitness'] >= -EPSILON:
                return record_and_return(success=True)

        # (mu, lambda) selection: re-sort after local search, keep top MU
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


# 5. EXPERIMENTAL PROTOCOL — 30 RUNS
def run_experiment(label="Memetic"):
    from es import analyse_convexity

    seeds   = list(range(BASE_SEED, BASE_SEED + N_RUNS))
    results = []

    print(f"\n{'='*58}")
    print(f"{label} (mu={MU}, lambda={LAM}, k={K_LOCAL},"
          f" depth={DEPTH}, sigma_local={SIGMA_LOCAL})")
    print(f"Rastrigin d={N}  |  Budget: {MAX_EVALS:,}  |  eps = {EPSILON}")
    print(f"{'='*58}")

    for i, seed in enumerate(seeds):
        res = run_memetic(seed)

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
    cost_runs = [-r['best_fitness'] for r in results]
    runs_ok   = [r for r in results if r['success']]
    runs_fail = [r for r in results if not r['success']]
    evals_ok  = [r['n_evals'] for r in runs_ok]

    sr      = len(runs_ok) / N_RUNS
    mbf     = np.mean(cost_runs)
    mbf_std = np.std(cost_runs)
    aes     = np.mean(evals_ok) if runs_ok else None
    aes_std = np.std(evals_ok)  if runs_ok else None

    # aggregated convexity
    nc_total  = [r['conv_fraction']  for r in results   if not np.isnan(r['conv_fraction'])]
    nc_ok     = [r['conv_fraction']  for r in runs_ok   if not np.isnan(r['conv_fraction'])]
    nc_fail   = [r['conv_fraction']  for r in runs_fail if not np.isnan(r['conv_fraction'])]
    int_total = [r['conv_intensity'] for r in results   if not np.isnan(r['conv_intensity'])]

    conv_mean   = np.mean(nc_total)  if nc_total  else np.nan
    conv_ok_m   = np.mean(nc_ok)     if nc_ok     else np.nan
    conv_fail_m = np.mean(nc_fail)   if nc_fail   else np.nan
    intens_mean = np.mean(int_total) if int_total else np.nan

    # extended SR: formal success OR failure with convex curve (nc < NC_THRESHOLD)
    runs_conv     = [r for r in runs_fail
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
    print(f"    Success = {conv_ok_m:.3f}")
    print(f"    Failure = {conv_fail_m:.3f}")
    print(f"    Mean intensity = {intens_mean:.4f}")

    return {
        'label': label, 'results': results, 'seeds': seeds,
        'sr': sr, 'sr_conv': sr_conv, 'n_conv': len(runs_conv),
        'mbf': mbf, 'mbf_std': mbf_std,
        'aes': aes, 'aes_std': aes_std, 'n_success': len(runs_ok),
        'conv_mean': conv_mean, 'conv_ok': conv_ok_m, 'conv_fail': conv_fail_m,
        'intens_mean': intens_mean,
    }



# 6. VISUALISATIONS
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
    import matplotlib
    matplotlib.use('Agg')
    from es import run_experiment as run_es_exp
    from es import plot_convergence as plot_conv_es
    from es import plot_sigma as plot_sigma_es

    exp_es  = run_es_exp(label="ES")
    exp_mem = run_experiment(label="Memetic")

    # comparison table
    print("\n\n" + "="*58)
    print("COMPARISON — ES vs Memetic  |  Rastrigin d=10")
    print("="*58)
    rows = []
    for exp in [exp_es, exp_mem]:
        rows.append({
            "Algorithm":  exp['label'],
            "SR (%)":     f"{exp['sr']*100:.1f}",
            "SR_conv (%)":f"{exp['sr_conv']*100:.1f}",
            "MBF":        f"{exp['mbf']:.4f}",
            "Std MBF":    f"{exp['mbf_std']:.4f}",
            "AES":        f"{exp['aes']:.0f}" if exp['aes'] else "N/A",
            "Std AES":    f"{exp['aes_std']:.0f}" if exp['aes_std'] else "N/A",
            "NC total":   f"{exp['conv_mean']:.3f}",
            "NC success": f"{exp['conv_ok']:.3f}",
            "NC failure": f"{exp['conv_fail']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    colors = {"ES": "#5f7fbf", "Memetic": "#bf5f5f"}

    # figure 1a: ES convergence
    fig1a, ax1a = plt.subplots(figsize=(9, 5))
    plot_conv_es(exp_es, ax1a, color=colors["ES"])
    ax1a.axhline(y=-EPSILON, color="gray", linestyle="--",
                 linewidth=1, label=f"-eps = {-EPSILON}")
    ax1a.set_xlabel("J(x) evaluations")
    ax1a.set_ylabel("Best F(x) = -J(x)")
    ax1a.set_title("Convergence — ES  |  Rastrigin d=10")
    ax1a.legend(); ax1a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_es.png", dpi=150, bbox_inches="tight")

    # figure 1b: Memetic convergence
    fig1b, ax1b = plt.subplots(figsize=(9, 5))
    plot_convergence(exp_mem, ax1b, color=colors["Memetic"])
    ax1b.axhline(y=-EPSILON, color="gray", linestyle="--",
                 linewidth=1, label=f"-eps = {-EPSILON}")
    ax1b.set_xlabel("J(x) evaluations")
    ax1b.set_ylabel("Best F(x) = -J(x)")
    ax1b.set_title("Convergence — Memetic  |  Rastrigin d=10")
    ax1b.legend(); ax1b.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_memetic.png", dpi=150, bbox_inches="tight")

    # figure 2: box plots
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    data = [[-r['best_fitness'] for r in exp_es['results']],
            [-r['best_fitness'] for r in exp_mem['results']]]
    bp = ax2.boxplot(data, tick_labels=["ES", "Memetic"], patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp['boxes'], [colors["ES"], colors["Memetic"]]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax2.set_ylabel("Best J found")
    ax2.set_title("MBF — 30 runs  |  Rastrigin d=10")
    ax2.set_yscale("log"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_boxplot.png", dpi=150, bbox_inches="tight")

    # figure 3: sigma trajectory
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    plot_sigma_es(exp_es, ax3, color=colors["ES"])
    plot_sigma(exp_mem, ax3, color=colors["Memetic"])
    ax3.set_xlabel("Generation"); ax3.set_ylabel("Mean sigma")
    ax3.set_title("Step-size trajectory — ES vs Memetic  |  Rastrigin d=10")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_sigma.png", dpi=150, bbox_inches="tight")

    print("\nFigures saved:")
    print("  convergence_es.png")
    print("  convergence_memetic.png")
    print("  comparison_boxplot.png")
    print("  comparison_sigma.png")
