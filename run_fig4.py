import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import problem_generation, offline_data_generation, repeated_offline_data_generation, find_toff, compute_d_e
from src.algorithms import OOPE

def evaluate_gap(d_eff, T, non_zero_arm, offline_frac_dict, A, d, optimal_arm, theta, V_pi_o):
    toff = find_toff(d_eff, T, V_pi_o)
    offline_data_dict = repeated_offline_data_generation(toff, non_zero_arm, offline_frac_dict, theta, A)
    
    oope_val = OOPE(T, A, d, non_zero_arm, toff, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict, use_fw=False)
    oopefw_val = OOPE(T, A, d, non_zero_arm, toff, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict, use_fw=True)
    
    return toff, T, (oope_val - oopefw_val)

if __name__ == "__main__":
    dimensions = [17, 20, 22]
    T_horizons =[2000, 20000, 100000]
    n_runs = 15
    
    for d in dimensions:
        K = d**2
        n_support = 3 * d
        
        theta, A, optimal_arm, _ = problem_generation(d, K)
        # Create base offline partition to fix the eigenspectrum 
        offline_frac_dict, _, V_pi_o, non_zero_arm = offline_data_generation(500000, n_support, theta, A, d)
        
        d_eff_range = np.arange(1, d, 1.3)
        jobs =[(d_eff, T) for T in T_horizons for d_eff in d_eff_range for _ in range(n_runs)]
        
        print(f"\n--- Running evaluation for dimension {d} ---")
        worker = partial(evaluate_gap, non_zero_arm=non_zero_arm, offline_frac_dict=offline_frac_dict, 
                         A=A, d=d, optimal_arm=optimal_arm, theta=theta, V_pi_o=V_pi_o)
                         
        results = Parallel(n_jobs=-1, verbose=5)(delayed(worker)(d_eff, T) for d_eff, T in jobs)

        plt.figure(figsize=(10, 6))
        
        colors = {2000: 'r', 20000: 'b', 100000: 'orange'}
        for T in T_horizons:
            diffs_for_T = [res[2] for res in results if res[1] == T]
            
            # Group by d_eff (mapped uniquely to toffs for this T)
            unique_d_effs = []
            mean_diffs =[]
            
            # Manually extract the unique toff points mapped from d_eff_range
            idx = 0
            for d_eff in d_eff_range:
                batch = diffs_for_T[idx : idx + n_runs]
                idx += n_runs
                unique_d_effs.append(d_eff)
                mean_diffs.append(np.mean(batch))
                
            plt.plot(unique_d_effs, mean_diffs, marker='o', linestyle='-', color=colors[T], label=f'T={int(T/1000)}k')

        plt.axhline(0, color='grey', linestyle='--', linewidth=1)
        plt.title(f'Performance gap between OOPE and OOPE-FW for d={d}')
        plt.xlabel('$d_{eff}$')
        plt.ylabel('$\Delta$ Regret')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'figure4_d{d}.png', dpi=300)
        plt.show()