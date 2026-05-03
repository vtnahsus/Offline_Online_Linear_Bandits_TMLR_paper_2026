import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from scipy.stats import t

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import problem_generation, offline_data_generation, repeated_offline_data_generation
from src.algorithms import OOPE

def run_oope_variants(T, use_fw, non_zero_arm, offline_frac_dict, A, d, optimal_arm, theta, V_pi_o, T_o):
    print("FW is",use_fw,"for online horizon",T)
    offline_data_dict = repeated_offline_data_generation(T_o, non_zero_arm, offline_frac_dict, theta, A)
    regret = OOPE(T, A, d, non_zero_arm, T_o, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict, use_fw=use_fw)
    return use_fw, T, regret

if __name__ == "__main__":
    d = 30
    K = 900
    T_o = 1000000
    n_support = 100
    n_runs = 50
    T_range = np.arange(1000, 10000, 1000)
    mode= "Uniform"

    theta, A, optimal_arm, _ = problem_generation(d, K, mode)
    offline_frac_dict, _, V_pi_o, non_zero_arm = offline_data_generation(T_o, n_support, theta, A, d)

    jobs =[(use_fw, T) for use_fw in [False, True] for T in T_range for _ in range(n_runs)]
    worker = partial(run_oope_variants, non_zero_arm=non_zero_arm, offline_frac_dict=offline_frac_dict, 
                     A=A, d=d, optimal_arm=optimal_arm, theta=theta, V_pi_o=V_pi_o, T_o=T_o)

    print(f"Running experiments in parallel on {os.cpu_count()} cores...")
    results = Parallel(n_jobs=-1, verbose=5)(delayed(worker)(use_fw, T) for use_fw, T in jobs)

    res_dicts = {False: {T:[] for T in T_range}, True: {T:[] for T in T_range}}
    for use_fw, T, r in results:
        res_dicts[use_fw][T].append(r)

    def get_stats(data):
        m, ci = [],[]
        for T in T_range:
            arr = np.array(data[T])
            m.append(np.mean(arr))
            ci.append(t.ppf(0.975, df=n_runs-1) * (np.std(arr) / np.sqrt(n_runs)) if n_runs > 1 else 0)
        return m, ci

    oope_m, oope_ci = get_stats(res_dicts[False])
    fw_m, fw_ci = get_stats(res_dicts[True])

    plt.figure(figsize=(10, 6))
    plt.plot(T_range, oope_m, marker='o', color='b', label='OOPE')
    plt.plot(T_range, fw_m, marker='o', color='r', label='OOPE-FW')

    plt.errorbar(T_range, oope_m, yerr=oope_ci, fmt='none', color='b', capsize=5)
    plt.errorbar(T_range, fw_m, yerr=fw_ci, fmt='none', color='r', capsize=5)

    plt.title('Comparison between OOPE and OOPE-FW')
    plt.xlabel('Online Horizon ($T$)')
    plt.ylabel('Average Cumulative Regret')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(T_range,[f'{int(x/1000)}k' for x in T_range])
    plt.savefig('figure3.png', dpi=300)
    plt.show()
