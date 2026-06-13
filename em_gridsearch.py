"""
Grid Search — Memetic Algorithm (mu, lambda) + greedy local search
Benchmark function: Rastrigin (D=10 and D=15)

Parameters swept:
  mu     -- number of parents
  lam    -- number of offspring
  sigma0 -- initial step size

Fixed parameters (same as em_simples.py):
  local search: K=3, depth=50, sigma_local=0.5
  budget: 100,000 evaluations
  30 runs per configuration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from es import analyse_convexity

# FIXED PARAMETERS

N            = 15
SIGMA_MIN    = 1e-5
X_MIN        = -5.12
X_MAX        =  5.12
MAX_EVALS    = 100_000
EPSILON      = 0.05
NC_THRESHOLD = 0.20
N_RUNS       = 30
BASE_SEED    = 42

K_LOCAL      = 3
DEPTH        = 50
SIGMA_LOCAL  = 0.5


# PARAMETER GRID
GRID = {
    'mu':     [50, 70, 100],
    'lam':    [200, 400, 600],
    'sigma0': [0.5, 1.0, 2.0],
}


# BASE FUNCTIONS (replicated here for dynamic parametrisation)
def rastrigin(x):
    A = 10.0
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def fitness(x):
    return -rastrigin(x)

def initialise_population(mu, sigma0, rng):
    return [{'x': rng.uniform(X_MIN, X_MAX, N),
             'sigma': np.full(N, sigma0),
             'fitness': -np.inf}
            for _ in range(mu)]

def recombine(parent1, parent2, rng):
    mask = rng.random(N) < 0.5
    return {'x':       np.where(mask, parent1['x'], parent2['x']),
            'sigma':   0.5 * (parent1['sigma'] + parent2['sigma']),
            'fitness': -np.inf}

def mutate(ind, g1, g2, rng):
    s = ind['sigma'] * np.exp(g1 * rng.standard_normal()
                              + g2 * rng.standard_normal(N))
    s = np.maximum(s, SIGMA_MIN)
    x = np.clip(ind['x'] + s * rng.standard_normal(N), X_MIN, X_MAX)
    return {'x': x, 'sigma': s, 'fitness': -np.inf}

def local_search(ind, rng):
    x, j, ev = ind['x'].copy(), rastrigin(ind['x']), 0
    for _ in range(DEPTH):
        xc = np.clip(x + SIGMA_LOCAL * rng.standard_normal(N), X_MIN, X_MAX)
        jc = rastrigin(xc); ev += 1
        if jc < j:
            x, j = xc, jc
    return {'x': x, 'sigma': ind['sigma'].copy(), 'fitness': -j}, ev


# ONE MEMETIC RUN WITH DYNAMIC PARAMETERS
def run_memetic(seed, mu, lam, sigma0):
    g1 = 1.0 / np.sqrt(2 * N)
    g2 = 1.0 / np.sqrt(2 * np.sqrt(N))

    rng      = np.random.default_rng(seed)
    n_evals  = 0
    history_best  = []
    history_evals = []

    # initialise
    pop = initialise_population(mu, sigma0, rng)
    for ind in pop:
        ind['fitness'] = fitness(ind['x']); n_evals += 1

    best = max(pop, key=lambda i: i['fitness'])
    best = {'x': best['x'].copy(), 'sigma': best['sigma'].copy(),
            'fitness': best['fitness']}

    history_best.append(best['fitness'])
    history_evals.append(n_evals)

    while n_evals < MAX_EVALS:
        offspring_list = []

        # ES step
        for _ in range(lam):
            i1, i2 = rng.choice(mu, size=2, replace=False)
            child = mutate(recombine(pop[i1], pop[i2], rng), g1, g2, rng)
            child['fitness'] = fitness(child['x']); n_evals += 1
            offspring_list.append(child)
            if n_evals >= MAX_EVALS:
                break

        offspring_list.sort(key=lambda i: i['fitness'], reverse=True)

        # local search on K_LOCAL best
        for j in range(min(K_LOCAL, len(offspring_list))):
            if n_evals >= MAX_EVALS:
                break
            offspring_list[j], ev = local_search(offspring_list[j], rng)
            n_evals += ev
            if offspring_list[j]['fitness'] > best['fitness']:
                best = {'x': offspring_list[j]['x'].copy(),
                        'sigma': offspring_list[j]['sigma'].copy(),
                        'fitness': offspring_list[j]['fitness']}
            if best['fitness'] >= -EPSILON:
                history_best.append(best['fitness'])
                history_evals.append(n_evals)
                return {'success': True, 'best_fitness': best['fitness'],
                        'n_evals': n_evals,
                        'history_best':  history_best,
                        'history_evals': history_evals}

        offspring_list.sort(key=lambda i: i['fitness'], reverse=True)
        pop = offspring_list[:mu]

        if pop[0]['fitness'] > best['fitness']:
            best = {'x': pop[0]['x'].copy(), 'sigma': pop[0]['sigma'].copy(),
                    'fitness': pop[0]['fitness']}

        history_best.append(best['fitness'])
        history_evals.append(n_evals)

    return {'success': best['fitness'] >= -EPSILON,
            'best_fitness': best['fitness'],
            'n_evals': n_evals,
            'history_best':  history_best,
            'history_evals': history_evals}



# EVALUATE ONE CONFIGURATION — 30 RUNS
def evaluate_config(mu, lam, sigma0):
    results = []
    for i in range(N_RUNS):
        res = run_memetic(BASE_SEED + i, mu, lam, sigma0)

        nc_frac, intens = analyse_convexity(
            res['history_evals'], res['history_best'])
        res['conv_fraction']  = nc_frac
        res['conv_intensity'] = intens
        results.append(res)

    cost_runs = [-r['best_fitness'] for r in results]
    runs_ok   = [r for r in results if r['success']]
    runs_fail = [r for r in results if not r['success']]
    evals_ok  = [r['n_evals'] for r in runs_ok]

    sr      = len(runs_ok) / N_RUNS
    mbf     = float(np.mean(cost_runs))
    mbf_std = float(np.std(cost_runs))
    aes     = float(np.mean(evals_ok)) if runs_ok else None
    aes_std = float(np.std(evals_ok))  if runs_ok else None

    nc_vals    = [r['conv_fraction'] for r in results
                  if not np.isnan(r['conv_fraction'])]
    conv_mean  = float(np.mean(nc_vals)) if nc_vals else np.nan

    runs_conv = [r for r in runs_fail
                 if not np.isnan(r['conv_fraction'])
                 and r['conv_fraction'] < NC_THRESHOLD]
    sr_conv   = (len(runs_ok) + len(runs_conv)) / N_RUNS

    return {
        'mu': mu, 'lam': lam, 'sigma0': sigma0,
        'sr':      sr,
        'sr_conv': sr_conv,
        'mbf':     mbf,
        'mbf_std': mbf_std,
        'aes':     aes,
        'aes_std': aes_std,
        'conv_mean': conv_mean,
    }


# FULL GRID SEARCH
def run_grid_search():
    combinations = list(itertools.product(
        GRID['mu'], GRID['lam'], GRID['sigma0']))
    n_total = len(combinations)

    print(f"\n{'='*60}")
    print(f"Grid Search — Memetic  |  Rastrigin d={N}")
    print(f"Configurations: {n_total}  |  Runs per config: {N_RUNS}")
    print(f"Budget: {MAX_EVALS:,}  |  eps = {EPSILON}  |  NC_THRESHOLD = {NC_THRESHOLD}")
    print(f"{'='*60}\n")

    rows = []
    for k, (mu, lam, sigma0) in enumerate(combinations, 1):
        print(f"[{k:02d}/{n_total}] mu={mu:3d}  lam={lam:3d}  sigma0={sigma0:.1f} ... ",
              end='', flush=True)
        row = evaluate_config(mu, lam, sigma0)
        rows.append(row)
        aes_str = f"{row['aes']:.0f}" if row['aes'] else 'N/A'
        print(f"SR={row['sr']*100:.1f}%  SR_conv={row['sr_conv']*100:.1f}%"
              f"  MBF={row['mbf']:.4f}  AES={aes_str}")

    df = pd.DataFrame(rows)
    df = df.sort_values('sr_conv', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = run_grid_search()

    print(f"\n{'='*60}")
    print("RANKING — Top 5 configurations by SR_conv")
    print(f"{'='*60}")
    cols = ['mu', 'lam', 'sigma0', 'sr', 'sr_conv', 'mbf', 'mbf_std', 'aes', 'conv_mean']
    print(df[cols].head(5).to_string(index=False, float_format='%.4f'))

    print(f"\n{'='*60}")
    print("RANKING — Top 5 configurations by MBF (lower is better)")
    print(f"{'='*60}")
    print(df.sort_values('mbf')[cols].head(5).to_string(index=False, float_format='%.4f'))

    df.to_csv('gridsearch_results.csv', index=False)
    print('\nFull table saved: gridsearch_results.csv')
