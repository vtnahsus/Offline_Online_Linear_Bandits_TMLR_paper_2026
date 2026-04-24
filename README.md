# Offline_Online_Linear_Bandits_TMLR_paper_2026
Code for experiments run in the Offline to Online in Linear Bandits paper.
# Regret Minimization in Linear Bandits with Offline Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of the paper:
**"Regret minimization in Linear Bandits with offline data via extended D-optimal exploration"** *(Accepted at TMLR)*.

It provides the code to reproduce all theoretical bounds and numerical experiments presented in the paper, including the **Offline-Online Phased Elimination (OOPE)** algorithm and its Frank-Wolfe variant (**OOPE-FW**).

## 📖 Abstract
We consider the problem of online regret minimization in stochastic linear bandits with access to prior observations (offline data). We introduce the **OOPE** algorithm, which effectively incorporates offline data to substantially reduce online regret by utilizing an extended D-optimal design. We also provide **OOPE-FW**, a Frank-Wolfe approximation to the extended optimal design, which improves the dimension dependence in the regret bound from $\mathcal{O}(d^2)$ to $\mathcal{O}(d^2 / d_{\text{eff}})$. 


## ⚙️ Installation
To run the code, you will need Python 3.8 or higher. We recommend using a virtual environment.
Clone the repository:

git clone https://github.com/vtnahsus/Offline_Online_Linear_Bandits_TMLR_paper_2026.git

cd Offline_Online_Linear_Bandits_TMLR_paper_2026

Install the required packages:
pip install -r requirements.txt

## 🚀 Reproducing the Experiments
The experiments use joblib for parallel execution to drastically speed up the Monte Carlo simulations. By default, they will use all available CPU cores (n_jobs=-1).
You can reproduce the figures from the paper by running the scripts directly from the root directory:

Figure 1: Improved performance with increasing offline data.
Evaluates how the regret of OOPE decreases as the offline data horizon ($T_{off}$) increases.
python scripts/run_fig1.py

Figure 2: Comparison against Baselines
Compares the performance of OOPE against warm-started LinUCB and warm-started LinTS.
python scripts/run_fig2.py

Figure 3: OOPE vs. OOPE-FW
Compares the standard OOPE algorithm with its Frank-Wolfe variant (OOPE-FW) in a setting with small effective dimension and a large number of arms.
python scripts/run_fig3.py

Figure 4: Performance gap across Effective Dimensions ($d_{eff})$.
Calculates the $\Delta$ Regret between OOPE and OOPE-FW across different dimensions and horizons to show where the Frank-Wolfe approximation provides the most benefit.
python scripts/run_fig4.py

## ⚖️ License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🗂️ Repository Structure

The code is modularized for ease of use and readability:

```text
├── src/
│   ├── __init__.py
│   ├── utils.py          # Environment simulation, sub-optimality gaps, offline data generation
│   └── algorithms.py     # Core algorithms: OOPE, OOPE-FW, Warm-started LinUCB & LinTS
├── scripts/
│   ├── run_fig1.py       # Exp 1: Performance with increasing offline data
│   ├── run_fig2.py       # Exp 2: Comparison against LinUCB and LinTS baselines
│   ├── run_fig3.py       # Exp 3: Performance of OOPE vs OOPE-FW
│   └── run_fig4.py       # Exp 4: Regret difference over effective dimension (d_eff)
├── requirements.txt      # Python dependencies
└── README.md

