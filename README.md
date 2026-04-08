# Representational Drift: Synaptic Noise and Intrinsic Excitability

Computational neuroscience project by Elena Egorova and Sara Bezrukavnikov.

## Overview

We study how synaptic noise and slow intrinsic excitability fluctuations interact
to determine whether a Hebbian memory trace can be faithfully rescued by homeostatic
regulation after network suppression.

A minimal rate-based recurrent network (N=10) with an all-inhibitory structural
baseline undergoes a three-day protocol:

- **Day 1 (15 s)** — encoding: sparse input activates 5 neurons, Hebbian plasticity
  builds a weight increment delta_W
- **Day 2 (5 s)** — suppression: uniform negative input I_p = -3 drives all rates
  below zero via ReLU, silencing Hebbian updates; passive weight decay toward W_struct
- **Day 3 (20 s)** — homeostatic rescue: I_p remains active, homeostatic drive h(t)
  rises to overcome suppression and restore mean firing rates

Four conditions share an identical Day 1 and diverge from Day 2 onward:

| Condition | Synaptic noise on dW | OU excitability eps_i |
|-----------|---------------------|----------------------|
| Baseline  | no  | no  |
| A         | yes | no  |
| B         | no  | yes |
| C         | yes | yes |

## Model equations

    tau_r * dr/dt = -r + W * phi(r) + I + eps + h
    tau_W * dW/dt = -lambda * (W - W_struct) + eta * phi(r) * phi(r)^T
    tau_h * dh/dt = kappa * (m* - mean(r))
    tau_eps * d_eps_i = -eps_i * dt + sigma * sqrt(2*dt/tau_eps) * xi_i

phi = ReLU, W_struct entries drawn from -Uniform(0, W_s/N), diagonal = 0.

## Key result

Synaptic noise alone (A) and excitability alone (B) preserve memory fidelity.
Their combination (C) causes catastrophic failure: the network converges to a
wrong attractor despite homeostatic rate recovery (m_mem = 0.08 vs 1.11 in Baseline).

## Repository structure

```
src/
  network.py           — HebbianNetwork class (rate dynamics, Hebbian weights, homeostasis)
  generate_inputs.py   — generates sparse input patterns, saves to data/

notebooks/
  memory_homeostasis.ipynb  — full simulation: Days 1-3, all conditions, all figures

data/
  I_base.npy           — sparse encoding input, shape (10,)
  I_m.npy              — OU-drifting input sequence, shape (24, 10)

figures/               — all generated plots (produced by notebook)

report/
  report.tex           — LaTeX source
  report.pdf           — compiled report
  refs.bib             — bibliography
```

## Setup

```bash
conda create -n repr-drift python=3.11 -y
conda activate repr-drift
pip install -r requirements.txt

# optionally regenerate input data (already saved in data/)
python src/generate_inputs.py

# register kernel for Jupyter
python -m ipykernel install --user --name repr-drift
```

Open `notebooks/memory_homeostasis.ipynb` with the `repr-drift` kernel and run all cells.

## References

- Rule et al. (2019) Current Opinion in Neurobiology
- Qin et al. (2023) Nature Neuroscience
- Delamare et al. (2023) eLife
- Darshan & Rivkind (2022) Cell Reports
- Bauer et al. (2024) Nature Communications
