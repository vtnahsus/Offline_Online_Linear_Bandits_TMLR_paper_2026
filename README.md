# Offline_Online_Linear_Bandits_TMLR_paper_2026
Code for experiments run in the Offline to Online in Linear Bandits paper.
# Regret Minimization in Linear Bandits with Offline Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:
**"Regret minimization in Linear Bandits with offline data via extended D-optimal exploration"** *(Under review at TMLR)*.

It provides the code to reproduce all theoretical bounds and numerical experiments presented in the paper, including the **Offline-Online Phased Elimination (OOPE)** algorithm and its Frank-Wolfe variant (**OOPE-FW**).

## 📖 Abstract
We consider the problem of online regret minimization in stochastic linear bandits with access to prior observations (offline data). We introduce the **OOPE** algorithm, which effectively incorporates offline data to substantially reduce online regret by utilizing an extended D-optimal design. We also provide **OOPE-FW**, a Frank-Wolfe approximation to the extended optimal design, which improves the dimension dependence in the regret bound from $\mathcal{O}(d^2)$ to $\mathcal{O}(d^2 / d_{\text{eff}})$. 

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
