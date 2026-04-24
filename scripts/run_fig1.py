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

def run_experiment(toff, T, non_zero_arm, offline_frac_dict, A, d, optimal_arm, theta, V_pi_o):
    offline_data_dict = repeated_offline_data_generation(toff, non_zero_arm, offline_frac_dict)
    regret = OOPE(T, A, d, non_zero_arm, toff, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict)
    return toff, regret

def run_pure_online(T, A, d, optimal_arm, theta):
    # Pure online is just OOPE with Toff=0
    regret = OOPE(T, A, d, np.array([]), 0, optimal_arm, theta, np.zeros((d,d)), {}, {})
    return 0, regret

if __name__ == "__main__":
    d = 20
    K = 40
    T = 10000
    T_off_max = 100000
    n_support = 40
    n_runs = 50 
    mode= "Uniform"
    
    print("Generating Environment for Figure 1...")
    theta, A, optimal_arm, _ = problem_generation(d, K, mode)
    # Generate fixed partition using the largest Toff
    offline_frac_dict, _, V_pi_o, non_zero_arm = offline_data_generation(T_off_max, n_support, theta, A, d)

    T_off_range = np.arange(0, 100000, 10000) # includes 0 for pure online baseline
    jobs =[toff for toff in T_off_range for _ in range(n_runs)]
    
    print(f"Running experiments in parallel on {os.cpu_count()} cores...")
    worker = partial(run_experiment, T=T, non_zero_arm=non_zero_arm, offline_frac_dict=offline_frac_dict, 
                     A=A, d=d, optimal_arm=optimal_arm, theta=theta, V_pi_o=V_pi_o)
                     
    results = Parallel(n_jobs=-1, verbose=5)(delayed(worker)(toff) for toff in jobs)

    res_dict = {toff:[] for toff in T_off_range}
    for toff, r in results:
        res_dict[toff].append(r)

    means, cis = [],[]
    for toff in T_off_range:
        arr = np.array(res_dict[toff])
        means.append(np.mean(arr))
        cis.append(t.ppf(0.975, df=n_runs-1) * (np.std(arr) / np.sqrt(n_runs)) if n_runs > 1 else 0)

    # Separate baseline (toff=0) from OOPE results
    online_mean = means[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_off_range[1:], means[1:], marker='o', color='b', label='OOPE')
    plt.errorbar(T_off_range[1:], means[1:], yerr=cis[1:], fmt='none', color='b', capsize=5)
    
    # Pure online baseline line
    plt.axhline(y=online_mean, color='r', linestyle='--', label='Pure online baseline')
    
    plt.title('Average Regret vs. Offline Data Horizon ($T_{off}$)')
    plt.xlabel('Offline Horizon ($T_{off}$)')
    plt.ylabel('Average Cumulative Regret')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xticks(T_off_range[1:],[f'{int(x/1000)}k' for x in T_off_range[1:]])
    plt.savefig('figure1.png', dpi=300)
    plt.show()
