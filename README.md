# Numerical Stiffness in Biological Neural Networks: The *Drosophila melanogaster* Connectome

> *Rigidez numérica em redes neurais biológicas — o conectoma da Drosophila melanogaster como estudo de caso*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/drosophila-connectome-stiffness/blob/main/drosophila_connectome_stiffness.ipynb)

**Course:** Numerical Calculus (*Cálculo Numérico*) — ICT / UNIFESP  
**Repository:** `drosophila-connectome-stiffness`

## Authors

- Bruno de Abreu Correia
- Felipe Scarpin Mesquita
- Guilherme Pombo Takayama Augusto
- Júlia Cristina Perroni de Martini Pereira

## Objectives

This project investigates numerical stiffness arising in biological neural networks using real synaptic connectivity data from the *Drosophila melanogaster* central brain (`hemibrain:v1.2.1` via NeuPrint):

- **(a) ODE System Construction:** Build the linear system of Ordinary Differential Equations (ODEs) from real synaptic connectivity data.
- **(b) Spectral & Conditioning Analysis:** Relate the eigenvalues and condition number $\kappa_2(A)$ of the system matrix to numerical stiffness (stiffness ratio $S$ and critical explicit time step $h_{\text{crit}}$).
- **(c) Explicit vs. Implicit Integrators:** Compare explicit (Explicit Euler, RK4) and implicit (Implicit Euler, Implicit Trapezoidal) integration methods in terms of stability, error, and computational cost.
- **(d) Analytical Validation:** Validate the numerical integrators against the exact analytical solution (matrix exponential `scipy.linalg.expm`) on a small linear subsystem.
- **(e) Synthetic Network Control:** Compare the numerical stiffness of the biological connectome against synthetic Erdős–Rényi control networks of matching size and density.

## Project Structure

```text
drosophila-connectome-stiffness/
├── drosophila_connectome_stiffness.ipynb   # Main Google Colab notebook (planned)
├── data/                                   # Extracted connectome matrices (planned)
├── results/                                # Generated figures and LaTeX/CSV tables (planned)
├── report/                                 # Overleaf LaTeX report template (main.tex, referencias.bib)
├── UNIFESP_CN_PROPOSTA_PROJETO_FINAL.pdf   # Official project proposal
├── .gitignore
└── README.md
```

## Setup & Environment

1. Store your NeuPrint API token in a local `.env` file (ignored by Git):

   ```env
   NEUPRINT_TOKEN=your_token_here
   ```

2. Open the main notebook in Google Colab (once pushed to GitHub, replace `OWNER` in the Colab badge URL above with your GitHub username/organization) or run the analysis locally with Python (`numpy`, `scipy`, `matplotlib`, `neuprint-python`).
