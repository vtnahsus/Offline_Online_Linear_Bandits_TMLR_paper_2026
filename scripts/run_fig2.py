import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from scipy.stats import t

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import problem_generation, offline_data_generation, repeated_offline_data_generation
from src.algorithms import OOPE, LinUCB_warm_start, LinTS_warm_start

def run_algo(algo_name, T, non_zero_arm, offline_frac_dict, A, d, optimal_arm, theta, V_pi_o, T_o):
    offline_data_dict = repeated_offline_data_generation(T_o, non_zero_arm, offline_frac_dict, theta, A)
    if algo_name == "OOPE":
        r = OOPE(T, A, d, non_zero_arm, T_o, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict)
    elif algo_name == "LinUCB":
        r = LinUCB_warm_start(d, A, theta, optimal_arm, non_zero_arm, T_o, V_pi_o, offline_data_dict, T)
    else:
        r = LinTS_warm_start(d, A, optimal_arm, theta, non_zero_arm, offline_data_dict, T_o, V_pi_o, T)
    return algo_name, T, r

def calculate_stats(data_dict, T_range, n_runs):
    means, cis =[], []
    for T in T_range:
        arr = np.array(data_dict[T])
        means.append(np.mean(arr))
        cis.append(t.ppf(0.975, df=n_runs-1) * (np.std(arr) / np.sqrt(n_runs)) if n_runs > 1 else 0)
    return np.array(means), np.array(cis)

if __name__ == "__main__":
    d = 20
    K = 40
    T_o = 100000
    n_support = 40
    n_runs = 50
    T_range = np.arange(10000, 100000, 10000)

    theta, A, optimal_arm, _ = problem_generation(d, K)
    offline_frac_dict, _, V_pi_o, non_zero_arm = offline_data_generation(T_o, n_support, theta, A, d)

    jobs = [(algo, T) for algo in["OOPE", "LinUCB", "LinTS"] for T in T_range for _ in range(n_runs)]
    worker = partial(run_algo, non_zero_arm=non_zero_arm, offline_frac_dict=offline_frac_dict, 
                     A=A, d=d, optimal_arm=optimal_arm, theta=theta, V_pi_o=V_pi_o, T_o=T_o)

    print(f"Running experiments in parallel on {os.cpu_count()} cores...")
    results = Parallel(n_jobs=-1, verbose=5)(delayed(worker)(algo, T) for algo, T in jobs)

    res_dicts = {"OOPE": {T:[] for T in T_range}, "LinUCB": {T:[] for T in T_range}, "LinTS": {T:[] for T in T_range}}
    for algo, T, r in results:
        res_dicts[algo][T].append(r)

    oope_m, oope_ci = calculate_stats(res_dicts["OOPE"], T_range, n_runs)
    ucb_m, ucb_ci = calculate_stats(res_dicts["LinUCB"], T_range, n_runs)
    ts_m, ts_ci = calculate_stats(res_dicts["LinTS"], T_range, n_runs)

    plt.figure(figsize=(10, 6))
    plt.plot(T_range, oope_m, marker='o', color='b', label='OOPE')
    plt.plot(T_range, ucb_m, marker='o', color='red', label='LinUCB')
    plt.plot(T_range, ts_m, marker='o', color='green', label='LinTS')

    plt.errorbar(T_range, oope_m, yerr=oope_ci, fmt='none', color='b', capsize=5)
    plt.errorbar(T_range, ucb_m, yerr=ucb_ci, fmt='none', color='red', capsize=5)
    plt.errorbar(T_range, ts_m, yerr=ts_ci, fmt='none', color='green', capsize=5)

    plt.title('Comparison between OOPE, LinUCB and LinTS')
    plt.xlabel('Online Horizon ($T$)')
    plt.ylabel('Average Cumulative Regret')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(T_range,[f'{int(x/1000)}k' for x in T_range])
    plt.savefig('figure2.png', dpi=300)
    plt.show()
