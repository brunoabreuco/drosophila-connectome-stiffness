"""
Script da Parte 2 — Análise Espectral, Discos de Gershgorin, Condicionamento e Rigidez Numérica (Objetivo b).
Gera a Figura 2 (300 DPI) e exporta a Tabela 2 em CSV e LaTeX.
Projeto Final de Cálculo Numérico — ICT / UNIFESP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.connectome import build_system_matrix, load_connectome_data
from src.spectral import (
    compare_stabilization_modes,
    export_spectral_summary_tables,
    plot_eigenvalues_and_gershgorin,
)


def _print_mode_summary(label: str, res: dict) -> None:
    gersh = res["gershgorin"]
    print(f"\n--- {label} (N = {res['N']}) ---")
    print(
        f"  Gershgorin union bounds   : [{gersh['union_left_min']:.2f}, {gersh['union_right_max']:.2f}] s^-1 "
        f"(all eigs inside = {res['all_inside_gershgorin']})"
    )
    print(
        f"  Fastest decay |Re(lam_1)| : {res['fastest_decay_rate']:.4f} s^-1 "
        f"(tau_min = {res['tau_min_ms']:.4f} ms)"
    )
    print(
        f"  2nd fastest |Re(lam_2)|   : {res['second_fastest_decay_rate']:.4f} s^-1"
    )
    print(
        f"  Slowest decay |Re(lam_N)| : {res['slowest_decay_rate']:.4f} s^-1 "
        f"(tau_max = {res['tau_max_ms']:.2f} ms)"
    )
    print(
        f"  Spectral radius rho(A)    : {res['spectral_radius_rho']:.4f} s^-1"
    )
    print(
        f"  Max oscillatory |Im(lam)| : {res['max_imag_abs']:.4f} rad/s "
        f"({res['num_complex_eigenvalues']} complex eigs = {res['num_complex_pairs']} pairs, "
        f"{res['complex_fraction_pct']:.1f}%)"
    )
    print(
        f"  Stiffness Ratio S(A)      : {res['stiffness_ratio_S']:.4f}"
    )
    print(
        f"  SVD [sigma_min, sigma_max]: [{res['sigma_min']:.4f}, {res['sigma_max']:.4f}]"
    )
    print(
        f"  Condition Numbers         : kappa_2(A) = {res['cond_2_A']:.2f} | "
        f"kappa_inf(A) = {res['cond_inf_A']:.2f} | kappa_1(A) = {res['cond_1_A']:.2f}"
    )
    print(
        f"  Non-normality & Basis     : ||[A,A^T]||_F/||A||_F^2 = {res['non_normality_commutator']:.5f} | "
        f"Henrici = {res['henrici_departure']:.4f} | kappa_2(V) = {res['eigenvector_cond_2_V']:.2f}"
    )
    print(
        f"  Scaling correlations      : r(1+d_k, |Re(lam_k)|) = {res['corr_conductance_eigenvalue']:.4f} | "
        f"r(1+d_k, sigma_k) = {res['corr_conductance_singular_value']:.4f} "
        f"(median error = {res['median_rel_error_gershgorin_pct']:.2f}%)"
    )
    print(
        f"  Exact h_crit (Euler Expl.): {res['h_crit_euler']:.8f} s "
        f"(limiting eig = {res['h_crit_euler_limiting_eigenvalue'].real:.2f}"
        f"{res['h_crit_euler_limiting_eigenvalue'].imag:+.2f}j)"
    )
    print(
        f"  Exact h_crit (RK4)        : {res['h_crit_rk4']:.8f} s "
        f"(limiting eig = {res['h_crit_rk4_limiting_eigenvalue'].real:.2f}"
        f"{res['h_crit_rk4_limiting_eigenvalue'].imag:+.2f}j)"
    )


def main() -> None:
    print("=" * 80)
    print("PARTE 2 — ANÁLISE ESPECTRAL, DISCOS DE GERSHGORIN E CONDICIONAMENTO (OBJETIVO B)")
    print("=" * 80)

    W_n120, W_n20, neurons = load_connectome_data(data_dir=ROOT_DIR / "data")
    A_n120 = build_system_matrix(W_n120, epsilon=1.0, mode="laplacian")

    comp_120 = compare_stabilization_modes(W_n120, epsilon=1.0)
    comp_20 = compare_stabilization_modes(W_n20, epsilon=1.0)

    _print_mode_summary("SUB-REDE PRINCIPAL N=120 [MODE = LAPLACIAN]", comp_120["laplacian"])
    _print_mode_summary("SUB-REDE PRINCIPAL N=120 [MODE = SHIFT]", comp_120["shift"])
    _print_mode_summary("SUBSISTEMA REDUZIDO N=20 [MODE = LAPLACIAN]", comp_20["laplacian"])
    _print_mode_summary("SUBSISTEMA REDUZIDO N=20 [MODE = SHIFT]", comp_20["shift"])

    csv_path, tex_path = export_spectral_summary_tables(
        comp_120,
        comp_20,
        csv_path=ROOT_DIR / "results" / "tables" / "tab2_spectral_conditioning.csv",
        tex_path=ROOT_DIR / "results" / "tables" / "tab2_spectral_conditioning.tex",
    )
    print(f"\n[Tabelas exportadas]\n  - {csv_path}\n  - {tex_path}")

    fig_paths = plot_eigenvalues_and_gershgorin(
        W_n120,
        A_n120,
        neurons,
        save_paths=[
            ROOT_DIR / "results" / "figures" / "fig2_eigenvalues_gershgorin.png",
            ROOT_DIR / "report" / "fig2_eigenvalues_gershgorin.png",
        ],
        show=False,
    )
    print("[Figura 2 gerada em 300 DPI]")
    for path in fig_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
