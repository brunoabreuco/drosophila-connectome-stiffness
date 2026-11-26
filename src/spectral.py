"""
Módulo de análise espectral, Discos de Gershgorin, valores singulares (SVD),
condicionamento matricial, não-normalidade e limites críticos de estabilidade
para o conectoma da Drosophila melanogaster (Protocerebral Bridge — PB).

Projeto Final de Cálculo Numérico — ICT / UNIFESP.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.connectome import _neuron_class_color, build_system_matrix


def compute_gershgorin_disks(A: np.ndarray) -> Dict[str, Any]:
    """
    Calcula os centros e raios dos Discos de Gershgorin por linhas da matriz A:
        D_i = { z in C : |z - c_i| <= R_i },
    onde c_i = A_{ii} e R_i = sum_{j != i} |A_{ij}|.
    """
    A_arr = np.asarray(A, dtype=np.float64)
    centers = np.diag(A_arr).copy()
    abs_A = np.abs(A_arr)
    radii = np.sum(abs_A, axis=1) - np.abs(centers)
    left_bounds = centers - radii
    right_bounds = centers + radii

    return {
        "centers": centers,
        "radii": radii,
        "left_bounds": left_bounds,
        "right_bounds": right_bounds,
        "union_left_min": float(np.min(left_bounds)),
        "union_right_max": float(np.max(right_bounds)),
        "max_radius": float(np.max(radii)),
        "min_radius": float(np.min(radii)),
    }


def compute_rk4_stability_limit_for_eigenvalue(
    lam: complex,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> float:
    """
    Calcula o passo crítico exato h_crit > 0 de estabilidade absoluta do método
    Runge-Kutta clássico de 4ª ordem (RK4) para um autovalor complexo lam (Re(lam) < 0),
    resolvendo |R_RK4(h * lam)| = |1 + z + z^2/2 + z^3/6 + z^4/24| = 1 via bisseção.
    """
    if np.real(lam) >= 0.0:
        return 0.0

    mod_lam = float(np.abs(lam))
    lo = 0.0
    hi = 4.0 / mod_lam

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        z = mid * lam
        amp = abs(1.0 + z + 0.5 * (z ** 2) + (z ** 3) / 6.0 + (z ** 4) / 24.0)
        if amp <= 1.0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= tol * max(1.0, lo):
            break

    return float(lo)


def analyze_spectrum_and_conditioning(
    W: np.ndarray,
    A: np.ndarray,
    imag_tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Executa a análise espectral completa, verificação do Teorema de Gershgorin,
    decomposição em valores singulares (SVD), número de condicionamento,
    índice de não-normalidade e cálculo dos passos críticos exatos no plano complexo
    para Euler Explícito e Runge-Kutta de 4ª ordem (RK4).
    """
    W_arr = np.asarray(W, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)
    n = A_arr.shape[0]

    # 1. Discos de Gershgorin
    gershgorin = compute_gershgorin_disks(A_arr)
    centers = gershgorin["centers"]
    radii = gershgorin["radii"]

    # 2. Decomposição espectral A V = V Lambda
    eigenvalues_raw, eigenvectors_raw = np.linalg.eig(A_arr)
    sort_idx = np.argsort(np.real(eigenvalues_raw))  # Do mais negativo (rápido) ao menos negativo (lento)
    eigenvalues = eigenvalues_raw[sort_idx]
    eigenvectors = eigenvectors_raw[:, sort_idx]

    re_vals = np.real(eigenvalues)
    im_vals = np.imag(eigenvalues)
    re_abs = np.abs(re_vals)
    im_abs = np.abs(im_vals)
    mod_vals = np.abs(eigenvalues)

    # Verificação de inclusão nos Discos de Gershgorin: para cada autovalor lambda_k,
    # verifica se existe ao menos um disco D_i tal que |lambda_k - c_i| <= R_i + 1e-7.
    inclusion_flags: List[bool] = []
    for lam in eigenvalues:
        dists = np.abs(lam - centers)
        inclusion_flags.append(bool(np.any(dists <= radii + 1e-7)))
    all_inside_gershgorin = bool(all(inclusion_flags))

    # 3. Limites exatos de estabilidade absoluta no plano complexo para cada autovalor
    # Euler Explícito: |1 + h * lam_k| <= 1  <=>  h <= -2 * Re(lam_k) / |lam_k|^2
    h_ee_per_mode = -2.0 * re_vals / (mod_vals ** 2)
    idx_crit_ee = int(np.argmin(h_ee_per_mode))
    h_crit_euler_exact = float(h_ee_per_mode[idx_crit_ee])
    h_crit_euler_real_approx = float(2.0 / np.max(mod_vals))

    # RK4: |1 + z + z^2/2 + z^3/6 + z^4/24| <= 1 com z = h * lam_k
    h_rk4_per_mode = np.array(
        [compute_rk4_stability_limit_for_eigenvalue(complex(lam)) for lam in eigenvalues],
        dtype=np.float64,
    )
    idx_crit_rk4 = int(np.argmin(h_rk4_per_mode))
    h_crit_rk4_exact = float(h_rk4_per_mode[idx_crit_rk4])
    h_crit_rk4_real_approx = float(2.785293563405289 / np.max(mod_vals))

    # 4. Valores singulares e números de condicionamento
    singular_values = np.linalg.svd(A_arr, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    cond_2_A = float(sigma_max / sigma_min)
    cond_inf_A = float(np.linalg.cond(A_arr, p=np.inf))
    cond_1_A = float(np.linalg.cond(A_arr, p=1))

    # 5. Não-normalidade de A e assimetria direcional de W
    commutator = A_arr @ A_arr.T - A_arr.T @ A_arr
    norm_A_fro = float(np.linalg.norm(A_arr, ord="fro"))
    non_normality_commutator = float(
        np.linalg.norm(commutator, ord="fro") / (norm_A_fro ** 2)
    )
    henrici_departure = float(
        np.sqrt(max(0.0, (norm_A_fro ** 2) - float(np.sum(mod_vals ** 2)))) / norm_A_fro
    )
    eigenvector_cond_2_V = float(np.linalg.cond(eigenvectors, p=2))

    norm_W_fro = float(np.linalg.norm(W_arr, ord="fro"))
    synaptic_asymmetry_index = (
        float(np.linalg.norm(W_arr - W_arr.T, ord="fro") / norm_W_fro)
        if norm_W_fro > 0.0
        else 0.0
    )

    # 6. Escalonamento entre condutância biológica |c_(k)| = -A_{ii}, |Re(lambda_(k))| e sigma_(k)
    conductance_sorted = np.sort(-centers)[::-1]
    re_abs_sorted = np.sort(re_abs)[::-1]
    if float(np.std(conductance_sorted)) > 1e-12:
        corr_conductance_eigenvalue = float(
            np.corrcoef(conductance_sorted, re_abs_sorted)[0, 1]
        )
        corr_conductance_singular_value = float(
            np.corrcoef(conductance_sorted, singular_values)[0, 1]
        )
    else:
        corr_conductance_eigenvalue = 0.0
        corr_conductance_singular_value = 0.0
    median_rel_error_gershgorin_pct = float(
        np.median(np.abs(conductance_sorted - re_abs_sorted) / conductance_sorted) * 100.0
    )

    fastest_decay = float(np.max(re_abs))
    slowest_decay = float(np.min(re_abs))
    stiffness_ratio_S = float(fastest_decay / slowest_decay)
    spectral_radius_rho = float(np.max(mod_vals))
    spectral_abscissa_alpha = float(np.max(re_vals))

    num_complex = int(np.sum(im_abs > imag_tol))
    num_complex_pairs = num_complex // 2
    complex_fraction_pct = 100.0 * num_complex / float(n)

    return {
        "N": int(n),
        "eigenvalues_sorted": eigenvalues,
        "eigenvectors_sorted": eigenvectors,
        "singular_values": singular_values,
        "gershgorin": gershgorin,
        "all_inside_gershgorin": all_inside_gershgorin,
        "is_hurwitz_stable": bool(spectral_abscissa_alpha < 0.0),
        "fastest_decay_rate": fastest_decay,
        "second_fastest_decay_rate": float(re_abs_sorted[1]) if n > 1 else fastest_decay,
        "third_fastest_decay_rate": float(re_abs_sorted[2]) if n > 2 else fastest_decay,
        "slowest_decay_rate": slowest_decay,
        "second_slowest_decay_rate": float(re_abs_sorted[-2]) if n > 1 else slowest_decay,
        "spectral_radius_rho": spectral_radius_rho,
        "spectral_abscissa_alpha": spectral_abscissa_alpha,
        "stiffness_ratio_S": stiffness_ratio_S,
        "tau_min_ms": 1000.0 / fastest_decay,
        "tau_max_ms": 1000.0 / slowest_decay,
        "max_imag_abs": float(np.max(im_abs)),
        "num_complex_eigenvalues": num_complex,
        "num_complex_pairs": num_complex_pairs,
        "complex_fraction_pct": complex_fraction_pct,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "cond_2_A": cond_2_A,
        "cond_inf_A": cond_inf_A,
        "cond_1_A": cond_1_A,
        "cond_to_stiffness_ratio": float(cond_2_A / stiffness_ratio_S),
        "non_normality_commutator": non_normality_commutator,
        "henrici_departure": henrici_departure,
        "eigenvector_cond_2_V": eigenvector_cond_2_V,
        "synaptic_asymmetry_index": synaptic_asymmetry_index,
        "corr_conductance_eigenvalue": corr_conductance_eigenvalue,
        "corr_conductance_singular_value": corr_conductance_singular_value,
        "median_rel_error_gershgorin_pct": median_rel_error_gershgorin_pct,
        "h_crit_euler": h_crit_euler_exact,
        "h_crit_euler_real_approx": h_crit_euler_real_approx,
        "h_crit_euler_limiting_eigenvalue": complex(eigenvalues[idx_crit_ee]),
        "h_crit_rk4": h_crit_rk4_exact,
        "h_crit_rk4_real_approx": h_crit_rk4_real_approx,
        "h_crit_rk4_limiting_eigenvalue": complex(eigenvalues[idx_crit_rk4]),
        "h_ee_per_mode": h_ee_per_mode,
        "h_rk4_per_mode": h_rk4_per_mode,
        "conductance_sorted": conductance_sorted,
        "re_abs_sorted": re_abs_sorted,
    }


def compare_stabilization_modes(
    W: np.ndarray,
    epsilon: float = 1.0,
) -> Dict[str, Any]:
    """
    Compara as propriedades espectrais, de condicionamento e de rigidez numérica
    entre as duas formulações de estabilização da matriz de estado A:
      - mode='laplacian': A = -epsilon * I - D_in + W  (condutância biológica heterogênea)
      - mode='shift':     A = -gamma * I + W           (deslocamento espectral homogêneo)
    """
    A_lap = build_system_matrix(W, epsilon=epsilon, mode="laplacian")
    A_shift = build_system_matrix(W, epsilon=epsilon, mode="shift")

    res_lap = analyze_spectrum_and_conditioning(W, A_lap)
    res_shift = analyze_spectrum_and_conditioning(W, A_shift)

    gamma_shift = float(-A_shift[0, 0])

    return {
        "laplacian": res_lap,
        "shift": res_shift,
        "gamma_shift": gamma_shift,
        "stiffness_ratio_amplification_lap_vs_shift": float(
            res_lap["stiffness_ratio_S"] / res_shift["stiffness_ratio_S"]
        ),
        "cond_2_amplification_lap_vs_shift": float(
            res_lap["cond_2_A"] / res_shift["cond_2_A"]
        ),
    }


def export_spectral_summary_tables(
    comparison_n120: Dict[str, Any],
    comparison_n20: Dict[str, Any],
    csv_path: str | Path = "results/tables/tab2_spectral_conditioning.csv",
    tex_path: str | Path = "results/tables/tab2_spectral_conditioning.tex",
) -> Tuple[Path, Path]:
    """
    Exporta a tabela comparativa completa de métricas espectrais, de condicionamento,
    não-normalidade e passos críticos para N=120 e N=20 (modos 'laplacian' e 'shift')
    nos formatos CSV e LaTeX.
    """
    out_csv = Path(csv_path)
    out_tex = Path(tex_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    n120_lap = comparison_n120["laplacian"]
    n120_shf = comparison_n120["shift"]
    n20_lap = comparison_n20["laplacian"]
    n20_shf = comparison_n20["shift"]

    rows = [
        {
            "metric_key": "N",
            "metric_label_pt": "Dimensão do sistema ($N$)",
            "n120_laplacian": f"{n120_lap['N']}",
            "n20_laplacian": f"{n20_lap['N']}",
            "n120_shift": f"{n120_shf['N']}",
            "n20_shift": f"{n20_shf['N']}",
        },
        {
            "metric_key": "gershgorin_center_max_abs",
            "metric_label_pt": "Maior centro de Gershgorin $|c_{\\max}| = 1 + d_{\\max}$ ($\\text{s}^{-1}$)",
            "n120_laplacian": f"{np.max(np.abs(n120_lap['gershgorin']['centers'])):.2f}",
            "n20_laplacian": f"{np.max(np.abs(n20_lap['gershgorin']['centers'])):.2f}",
            "n120_shift": f"{np.max(np.abs(n120_shf['gershgorin']['centers'])):.2f}",
            "n20_shift": f"{np.max(np.abs(n20_shf['gershgorin']['centers'])):.2f}",
        },
        {
            "metric_key": "gershgorin_union_bounds",
            "metric_label_pt": "Intervalo real de Gershgorin $[\\min(c_i - R_i),\\, \\max(c_i + R_i)]$",
            "n120_laplacian": f"[{n120_lap['gershgorin']['union_left_min']:.1f}, {n120_lap['gershgorin']['union_right_max']:.1f}]",
            "n20_laplacian": f"[{n20_lap['gershgorin']['union_left_min']:.1f}, {n20_lap['gershgorin']['union_right_max']:.1f}]",
            "n120_shift": f"[{n120_shf['gershgorin']['union_left_min']:.1f}, {n120_shf['gershgorin']['union_right_max']:.1f}]",
            "n20_shift": f"[{n20_shf['gershgorin']['union_left_min']:.1f}, {n20_shf['gershgorin']['union_right_max']:.1f}]",
        },
        {
            "metric_key": "fastest_decay_rate",
            "metric_label_pt": "Modo mais rápido $\\max_k |\\text{Re}(\\lambda_k)|$ ($\\text{s}^{-1}$)",
            "n120_laplacian": f"{n120_lap['fastest_decay_rate']:.4f}",
            "n20_laplacian": f"{n20_lap['fastest_decay_rate']:.4f}",
            "n120_shift": f"{n120_shf['fastest_decay_rate']:.4f}",
            "n20_shift": f"{n20_shf['fastest_decay_rate']:.4f}",
        },
        {
            "metric_key": "slowest_decay_rate",
            "metric_label_pt": "Modo mais lento $\\min_k |\\text{Re}(\\lambda_k)|$ ($\\text{s}^{-1}$)",
            "n120_laplacian": f"{n120_lap['slowest_decay_rate']:.4f}",
            "n20_laplacian": f"{n20_lap['slowest_decay_rate']:.4f}",
            "n120_shift": f"{n120_shf['slowest_decay_rate']:.4f}",
            "n20_shift": f"{n20_shf['slowest_decay_rate']:.4f}",
        },
        {
            "metric_key": "spectral_radius_rho",
            "metric_label_pt": "Raio espectral $\\rho(A) = \\max_k |\\lambda_k|$ ($\\text{s}^{-1}$)",
            "n120_laplacian": f"{n120_lap['spectral_radius_rho']:.4f}",
            "n20_laplacian": f"{n20_lap['spectral_radius_rho']:.4f}",
            "n120_shift": f"{n120_shf['spectral_radius_rho']:.4f}",
            "n20_shift": f"{n20_shf['spectral_radius_rho']:.4f}",
        },
        {
            "metric_key": "max_imag_abs",
            "metric_label_pt": "Frequência oscilatória máxima $\\max_k |\\text{Im}(\\lambda_k)|$ ($\\text{rad/s}$)",
            "n120_laplacian": f"{n120_lap['max_imag_abs']:.2f}",
            "n20_laplacian": f"{n20_lap['max_imag_abs']:.2f}",
            "n120_shift": f"{n120_shf['max_imag_abs']:.2f}",
            "n20_shift": f"{n20_shf['max_imag_abs']:.2f}",
        },
        {
            "metric_key": "complex_eigenvalues",
            "metric_label_pt": "Autovalores complexos ($|\\text{Im}(\\lambda_k)| > 10^{-8}$)",
            "n120_laplacian": f"{n120_lap['num_complex_eigenvalues']} ({n120_lap['num_complex_pairs']} pares)",
            "n20_laplacian": f"{n20_lap['num_complex_eigenvalues']} ({n20_lap['num_complex_pairs']} pares)",
            "n120_shift": f"{n120_shf['num_complex_eigenvalues']} ({n120_shf['num_complex_pairs']} pares)",
            "n20_shift": f"{n20_shf['num_complex_eigenvalues']} ({n20_shf['num_complex_pairs']} pares)",
        },
        {
            "metric_key": "stiffness_ratio_S",
            "metric_label_pt": "Razão de rigidez $S(A) = \\max|\\text{Re}(\\lambda)| / \\min|\\text{Re}(\\lambda)|$",
            "n120_laplacian": f"{n120_lap['stiffness_ratio_S']:.2f}",
            "n20_laplacian": f"{n20_lap['stiffness_ratio_S']:.2f}",
            "n120_shift": f"{n120_shf['stiffness_ratio_S']:.2f}",
            "n20_shift": f"{n20_shf['stiffness_ratio_S']:.2f}",
        },
        {
            "metric_key": "sigma_max",
            "metric_label_pt": "Maior valor singular $\\sigma_{\\max}(A)$",
            "n120_laplacian": f"{n120_lap['sigma_max']:.2f}",
            "n20_laplacian": f"{n20_lap['sigma_max']:.2f}",
            "n120_shift": f"{n120_shf['sigma_max']:.2f}",
            "n20_shift": f"{n20_shf['sigma_max']:.2f}",
        },
        {
            "metric_key": "sigma_min",
            "metric_label_pt": "Menor valor singular $\\sigma_{\\min}(A)$",
            "n120_laplacian": f"{n120_lap['sigma_min']:.4f}",
            "n20_laplacian": f"{n20_lap['sigma_min']:.4f}",
            "n120_shift": f"{n120_shf['sigma_min']:.4f}",
            "n20_shift": f"{n20_shf['sigma_min']:.4f}",
        },
        {
            "metric_key": "cond_2_A",
            "metric_label_pt": "Número de condicionamento $\\kappa_2(A) = \\sigma_{\\max}/\\sigma_{\\min}$",
            "n120_laplacian": f"{n120_lap['cond_2_A']:.2f}",
            "n20_laplacian": f"{n20_lap['cond_2_A']:.2f}",
            "n120_shift": f"{n120_shf['cond_2_A']:.2f}",
            "n20_shift": f"{n20_shf['cond_2_A']:.2f}",
        },
        {
            "metric_key": "cond_inf_A",
            "metric_label_pt": "Número de condicionamento em norma infinito $\\kappa_\\infty(A)$",
            "n120_laplacian": f"{n120_lap['cond_inf_A']:.2f}",
            "n20_laplacian": f"{n20_lap['cond_inf_A']:.2f}",
            "n120_shift": f"{n120_shf['cond_inf_A']:.2f}",
            "n20_shift": f"{n20_shf['cond_inf_A']:.2f}",
        },
        {
            "metric_key": "non_normality_commutator",
            "metric_label_pt": "Índice de não-normalidade $\\|A A^T - A^T A\\|_F / \\|A\\|_F^2$",
            "n120_laplacian": f"{n120_lap['non_normality_commutator']:.5f}",
            "n20_laplacian": f"{n20_lap['non_normality_commutator']:.5f}",
            "n120_shift": f"{n120_shf['non_normality_commutator']:.5f}",
            "n20_shift": f"{n20_shf['non_normality_commutator']:.5f}",
        },
        {
            "metric_key": "henrici_departure",
            "metric_label_pt": "Afastamento de normalidade de Henrici $\\Delta_H(A) / \\|A\\|_F$",
            "n120_laplacian": f"{n120_lap['henrici_departure']:.4f}",
            "n20_laplacian": f"{n20_lap['henrici_departure']:.4f}",
            "n120_shift": f"{n120_shf['henrici_departure']:.4f}",
            "n20_shift": f"{n20_shf['henrici_departure']:.4f}",
        },
        {
            "metric_key": "eigenvector_cond_2_V",
            "metric_label_pt": "Condicionamento da base de autovetores $\\kappa_2(V)$ (Bauer-Fike)",
            "n120_laplacian": f"{n120_lap['eigenvector_cond_2_V']:.2f}",
            "n20_laplacian": f"{n20_lap['eigenvector_cond_2_V']:.2f}",
            "n120_shift": f"{n120_shf['eigenvector_cond_2_V']:.2f}",
            "n20_shift": f"{n20_shf['eigenvector_cond_2_V']:.2f}",
        },
        {
            "metric_key": "h_crit_euler",
            "metric_label_pt": "Passo crítico exato Euler Explícito $h_{\\text{crit, EE}}$ (s)",
            "n120_laplacian": f"{n120_lap['h_crit_euler']:.6f}",
            "n20_laplacian": f"{n20_lap['h_crit_euler']:.6f}",
            "n120_shift": f"{n120_shf['h_crit_euler']:.6f}",
            "n20_shift": f"{n20_shf['h_crit_euler']:.6f}",
        },
        {
            "metric_key": "h_crit_rk4",
            "metric_label_pt": "Passo crítico exato RK4 $h_{\\text{crit, RK4}}$ (s)",
            "n120_laplacian": f"{n120_lap['h_crit_rk4']:.6f}",
            "n20_laplacian": f"{n20_lap['h_crit_rk4']:.6f}",
            "n120_shift": f"{n120_shf['h_crit_rk4']:.6f}",
            "n20_shift": f"{n20_shf['h_crit_rk4']:.6f}",
        },
    ]

    fieldnames = [
        "metric_key",
        "metric_label_pt",
        "n120_laplacian",
        "n20_laplacian",
        "n120_shift",
        "n20_shift",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tex_lines = [
        "% Tabela gerada automaticamente por scripts/spectral_analysis.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Síntese espectral, condicionamento, não-normalidade e passos críticos de estabilidade para a sub-rede principal ($N = 120$) e de validação ($N = 20$) nos modelos Laplaciano ($A = -\\epsilon I - D_{\\text{in}} + W$) e de deslocamento homogêneo ($A = -\\gamma I + W$).}",
        "\\label{tab:espectral_condicionamento}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4.5pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "\\textbf{Métrica / Propriedade Numérica} & \\textbf{$N = 120$ (Lapl.)} & \\textbf{$N = 20$ (Lapl.)} & \\textbf{$N = 120$ (Shift)} & \\textbf{$N = 20$ (Shift)} \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['metric_label_pt']} & {row['n120_laplacian']} & {row['n20_laplacian']} & "
            f"{row['n120_shift']} & {row['n20_shift']} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    out_tex.write_text("\n".join(tex_lines), encoding="utf-8")

    return out_csv, out_tex


def plot_eigenvalues_and_gershgorin(
    W: np.ndarray,
    A: np.ndarray,
    neurons: List[Dict[str, Any]],
    save_paths: Optional[Sequence[str | Path] | str | Path] = None,
    show: bool = False,
) -> List[Path]:
    """
    Gera a Figura 2 (300 DPI, qualidade de publicação em português):
      - Painel (a): Espectro de autovalores sigma(A) no plano complexo sobreposto aos
                    Discos de Gershgorin D_i (N = 120), destacando os dois modos ultra-rápidos
                    dos mega-hubs LPsP, o aglomerado oscilatório complexo (Delta7/EPG/PEN)
                    e um zoom (inset) nos modos complexos e nos modos lentos próximos a -epsilon = -1,0.
      - Painel (b): Relação de escalonamento entre a condutância sináptica biológica de entrada
                    |c_(k)| = 1 + d_(k) (centros de Gershgorin), as taxas de decaimento modal
                    |Re(lambda_(k))| e os valores singulares sigma_(k)(A), evidenciando a
                    origem biológica da rigidez (S = 3722,86) e a amplificação do condicionamento
                    por não-normalidade (kappa_2 = 8159,90).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch

    if save_paths is None:
        target_paths: List[Path] = [
            Path("results/figures/fig2_eigenvalues_gershgorin.png"),
            Path("report/fig2_eigenvalues_gershgorin.png"),
        ]
    elif isinstance(save_paths, (str, Path)):
        target_paths = [Path(save_paths)]
    else:
        target_paths = [Path(p) for p in save_paths]

    analysis = analyze_spectrum_and_conditioning(W, A)
    eigs = analysis["eigenvalues_sorted"]
    re_eigs = np.real(eigs)
    im_eigs = np.imag(eigs)
    gersh = analysis["gershgorin"]
    centers = gersh["centers"]
    radii = gersh["radii"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11.3,
            "axes.labelsize": 10.3,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.4, 6.4), gridspec_kw={"width_ratios": [1.14, 1.08]}
    )
    fig.subplots_adjust(wspace=0.25, top=0.81, bottom=0.13, left=0.06, right=0.97)

    # =========================================================================
    # Painel (a): Espectro de A no Plano Complexo e Discos de Gershgorin D_i
    # =========================================================================
    order_din = np.argsort(centers)  # Do centro mais negativo (-3709) ao menos negativo (-14)

    # Seleciona discos representativos de todas as faixas de grau para visualização limpa
    rep_indices = [
        int(order_din[0]),    # LPsP Hub 1 (c = -3709, R = 3708)
        int(order_din[1]),    # LPsP Hub 2 (c = -3540, R = 3539)
        int(order_din[2]),    # Maior Delta7 (c = -1297, R = 1296)
        int(order_din[15]),   # Delta7 intermediário
        int(order_din[35]),   # Delta7 / EPG
        int(order_din[60]),   # EPG / PEN
        int(order_din[90]),   # PEN
        int(order_din[117]),  # ExR3 (c = -304)
        int(order_din[118]),  # OA-VPM3 (c = -60)
        int(order_din[119]),  # OA-VPM3 periférico (c = -14)
    ]

    for idx_pos, neuron_idx in enumerate(rep_indices):
        c_val = float(centers[neuron_idx])
        r_val = float(radii[neuron_idx])
        ntype = str(neurons[neuron_idx]["type"])
        col = _neuron_class_color(ntype)
        alpha_fill = 0.08 if idx_pos < 2 else 0.11
        lw = 1.25 if idx_pos < 3 else 0.9
        ls = "--" if idx_pos < 2 else "-"
        disk = Circle(
            (c_val, 0.0),
            r_val,
            facecolor=col,
             alpha=alpha_fill,
            edgecolor=col,
            linewidth=lw,
            linestyle=ls,
            zorder=2,
        )
        ax1.add_patch(disk)
        disk_border = Circle(
            (c_val, 0.0),
            r_val,
            facecolor="none",
            edgecolor=col,
            linewidth=lw,
            linestyle=ls,
            alpha=0.65,
            zorder=3,
        )
        ax1.add_patch(disk_border)

    # Plota os centros c_i = -1 - d_i no eixo real
    ax1.scatter(
        centers,
        np.zeros_like(centers),
        marker="+",
        s=28,
        color="#555555",
        alpha=0.55,
        linewidths=0.9,
        label="Centros $c_i = -1 - d_i$",
        zorder=4,
    )

    # Separa autovalores reais e complexos conjugados
    is_complex = np.abs(im_eigs) > 1e-8
    ax1.scatter(
        re_eigs[~is_complex],
        im_eigs[~is_complex],
        s=34,
        facecolor="#08519c",
        edgecolor="white",
        linewidth=0.5,
        label=f"Autovalores reais ({int(np.sum(~is_complex))})",
        zorder=5,
    )
    ax1.scatter(
        re_eigs[is_complex],
        im_eigs[is_complex],
        s=38,
        facecolor="#e6550d",
        edgecolor="white",
        linewidth=0.45,
        label=f"Pares complexos ({analysis['num_complex_pairs']} pares, $\\text{{Im}}\\neq 0$)",
        zorder=6,
    )

    # Destaca os 2 modos mais rápidos (LPsP) e o modo mais lento (-1.0)
    ax1.scatter(
        re_eigs[:2],
        im_eigs[:2],
        s=75,
        marker="D",
        facecolor="#d62728",
        edgecolor="black",
        linewidth=0.8,
        label="Modos rápidos LPsP ($\\lambda_1, \\lambda_2$)",
        zorder=7,
    )
    ax1.scatter(
        [re_eigs[-1]],
        [im_eigs[-1]],
        s=85,
        marker="*",
        facecolor="#2ca02c",
        edgecolor="black",
        linewidth=0.7,
        label="Modo lento de consenso ($\\lambda_{120} = -1,0$)",
        zorder=8,
    )

    # Linha vertical da abscissa espectral / ponto de tangência z = -epsilon = -1.0
    ax1.axvline(-1.0, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.85, zorder=2)
    ax1.axhline(0.0, color="#444444", linestyle="-", linewidth=0.6, alpha=0.5, zorder=1)

    # Anotações no Painel (a)
    ax1.annotate(
        f"Modos rápidos (Mega-hubs LPsP)\n"
        f"$\\lambda_1 = {re_eigs[0]:.1f}$, $\\lambda_2 = {re_eigs[1]:.1f}\\,\\mathrm{{s}}^{{-1}}$\n"
        f"($\\tau_{{\\min}} = {analysis['tau_min_ms']:.2f}\\,\\mathrm{{ms}}$)",
        xy=(re_eigs[0], 0.0),
        xytext=(-7520.0, 2400.0),
        fontsize=7.7,
        fontweight="bold",
        color="#7f0000",
        arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#fff5f0", edgecolor="#d62728", alpha=0.94),
        zorder=9,
    )

    ax1.annotate(
        f"Disco de Gershgorin $\\mathcal{{D}}_{{\\mathrm{{LPsP}}}}$\n"
        f"$c_1 = -3709,\\; R_1 = 3708$\n"
        f"Borda esq.: $c_1 - R_1 = -7417$",
        xy=(-7417.0, 0.0),
        xytext=(-7580.0, 1050.0),
        fontsize=7.6,
        color="#8c2d04",
        arrowprops=dict(arrowstyle="->", color="#8c2d04", lw=1.15),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#fd8d3c", alpha=0.94),
        zorder=9,
    )

    ax1.annotate(
        "Tangência comum de todos os discos\n"
        "$c_i + R_i = -\\epsilon = -1,0\\,\\mathrm{s}^{-1}$\n"
        "($\\lambda_{120} = -1,0000$, $\\tau_{\\max} = 1000\\,\\mathrm{ms}$)",
        xy=(-1.0, 0.0),
        xytext=(-3050.0, 3220.0),
        fontsize=7.6,
        fontweight="bold",
        color="#005a32",
        arrowprops=dict(arrowstyle="->", color="#005a32", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#f7fcf5", edgecolor="#2ca02c", alpha=0.94),
        zorder=9,
    )

    ax1.set_xlim(-7750.0, 450.0)
    ax1.set_ylim(-4150.0, 4150.0)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(
        "(a) Espectro $\\sigma(A)$ e Discos de Gershgorin $\\mathcal{D}_i$ ($N = 120$)",
        fontweight="bold",
        pad=14,
    )
    ax1.set_xlabel("Parte Real $\\mathrm{Re}(\\lambda)$ [$\\mathrm{s}^{-1}$] (Taxa de decaimento)")
    ax1.set_ylabel("Parte Imaginária $\\mathrm{Im}(\\lambda)$ [$\\mathrm{rad/s}$]")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(
        loc="lower right",
        fontsize=7.0,
        framealpha=0.95,
        borderpad=0.32,
        handletextpad=0.35,
    )

    # Inset no Painel (a): posicionado no semiplano inferior esquerdo sem tocar a legenda
    ax1_inset = ax1.inset_axes([0.11, 0.10, 0.36, 0.29])
    ax1_inset.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.6)
    ax1_inset.axvline(-1.0, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.9)
    ax1_inset.scatter(
        re_eigs[~is_complex],
        im_eigs[~is_complex],
        s=13,
        color="#08519c",
        alpha=0.85,
        zorder=3,
    )
    ax1_inset.scatter(
        re_eigs[is_complex],
        im_eigs[is_complex],
        s=17,
        color="#e6550d",
        alpha=0.92,
        zorder=4,
    )
    ax1_inset.scatter(
        [re_eigs[-1]],
        [0.0],
        s=46,
        marker="*",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    ax1_inset.set_xlim(-1440.0, 120.0)
    ax1_inset.set_ylim(-84.0, 84.0)
    ax1_inset.set_title(
        "Zoom: Núcleo Oscilatório ($\\Delta 7$/EPG/PEN)",
        fontsize=6.8,
        fontweight="bold",
        pad=3,
    )
    ax1_inset.set_xlabel("$\\mathrm{Re}(\\lambda)$ [$\\mathrm{s}^{-1}$]", fontsize=6.5, labelpad=1)
    ax1_inset.set_ylabel("$\\mathrm{Im}(\\lambda)$ [$\\mathrm{rad/s}$]", fontsize=6.5, labelpad=1)
    ax1_inset.tick_params(labelsize=6.1)
    ax1_inset.grid(True, linestyle=":", alpha=0.45)
    ax1_inset.annotate(
        f"$|\\mathrm{{Im}}|_{{\\max}} = {analysis['max_imag_abs']:.1f}$\n"
        f"(EPG$\\leftrightarrow$PEN)",
        xy=(-856.4, 56.73),
        xytext=(-650.0, 32.0),
        fontsize=5.8,
        color="#a63603",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#a63603", lw=0.75),
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.92, edgecolor="#fdd0a2"),
    )
    ax1_inset.annotate(
        "OA-VPM3 ($-12,9$)",
        xy=(-12.95, 0.0),
        xytext=(-760.0, -68.0),
        fontsize=5.9,
        color="#005a32",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#005a32", lw=0.75),
    )

    # =========================================================================
    # Painel (b): Escalonamento Condutância (1 + d_k) vs |Re(lambda_k)| vs sigma_k(A)
    # =========================================================================
    ranks = np.arange(1, len(eigs) + 1)
    conductance_sorted = analysis["conductance_sorted"]
    re_abs_sorted = analysis["re_abs_sorted"]
    sv_sorted = analysis["singular_values"]

    ax2.plot(
        ranks,
        conductance_sorted,
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        label="Grau de entrada $|c_{(k)}| = 1 + d_{(k)}$",
        zorder=3,
    )
    ax2.plot(
        ranks,
        re_abs_sorted,
        color="#d62728",
        linestyle="-",
        linewidth=2.1,
        label="Taxa de decaimento $|\\mathrm{Re}(\\lambda_{(k)})|$",
        zorder=5,
    )
    ax2.plot(
        ranks,
        sv_sorted,
        color="#1f77b4",
        linestyle="-.",
        linewidth=1.9,
        label="Valores singulares $\\sigma_{(k)}(A)$ (SVD)",
        zorder=4,
    )

    ax2.fill_between(
        ranks,
        re_abs_sorted,
        sv_sorted,
        color="#9ecae1",
        alpha=0.25,
        label="Diferença por assimetria ($A \\neq A^T$)",
    )

    ax2.set_yscale("log")
    ax2.set_xlim(0, 123)
    ax2.set_ylim(0.28, 7200.0)
    ax2.set_title(
        "(b) Escalonamento Grau Sináptico, Autovalores e Valores Singulares",
        fontweight="bold",
        pad=14,
    )
    ax2.set_xlabel("Índice ordenado do modo / posto $k$ ($1 = $ mais rápido $\\to 120 = $ mais lento)")
    ax2.set_ylabel("Magnitude espectral / singular [$\\mathrm{s}^{-1}$] (escala log)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.45)

    # Anotações de destaque no Painel (b)
    ax2.annotate(
        f"Mega-hubs LPsP ($k = 1, 2$)\n"
        f"$1 + d_1 = 3709\\,\\mathrm{{s}}^{{-1}}$\n"
        f"$|\\mathrm{{Re}}(\\lambda_1)| = {re_abs_sorted[0]:.2f}\\,\\mathrm{{s}}^{{-1}}$\n"
        f"$\\sigma_1(A) = {sv_sorted[0]:.2f}\\,\\mathrm{{s}}^{{-1}}$",
        xy=(1, re_abs_sorted[0]),
        xytext=(15, 2100.0),
        fontsize=8.0,
        fontweight="bold",
        color="#7f0000",
        arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fff5f0", edgecolor="#d62728", alpha=0.94),
    )

    ax2.annotate(
        f"Platô Anelar ($k = 3\\dots 117$)\n"
        f"$\\Delta 7$, EPG e PEN\n"
        f"Correlação $r = {analysis['corr_conductance_eigenvalue']:.4f}$\n"
        f"(desvio mediano ${analysis['median_rel_error_gershgorin_pct']:.2f}\\%$)",
        xy=(55, re_abs_sorted[54]),
        xytext=(34, 105.0),
        fontsize=8.0,
        color="#08306b",
        arrowprops=dict(arrowstyle="->", color="#08306b", lw=1.15),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f7fbff", edgecolor="#6baed6", alpha=0.94),
    )

    ax2.annotate(
        f"Modo Lento ($k = 120$) e Condicionamento\n"
        f"$|\\lambda_{{120}}| = \\epsilon = 1,0000\\,\\mathrm{{s}}^{{-1}} \\Rightarrow S(A) = {analysis['stiffness_ratio_S']:.2f}$\n"
        f"$\\sigma_{{120}}(A) = {sv_sorted[-1]:.4f}\\,\\mathrm{{s}}^{{-1}} \\Rightarrow \\kappa_2(A) = {analysis['cond_2_A']:.2f}$",
        xy=(120, sv_sorted[-1]),
        xytext=(36, 0.42),
        fontsize=8.0,
        fontweight="bold",
        color="#00441b",
        arrowprops=dict(arrowstyle="->", color="#00441b", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f7fcf5", edgecolor="#2ca02c", alpha=0.94),
    )

    # Caixa de resumo de estabilidade crítica para Parte 3
    ax2.text(
        0.03,
        0.37,
        f"Limite de Passo nos Métodos Explícitos:\n"
        f"• Euler Explícito: $h_{{\\mathrm{{crit, EE}}}} = {analysis['h_crit_euler']*1000:.3f}\\,\\mathrm{{ms}}$ ({analysis['h_crit_euler']:.6f} s)\n"
        f"• Runge-Kutta 4 (RK4): $h_{{\\mathrm{{crit, RK4}}}} = {analysis['h_crit_rk4']*1000:.3f}\\,\\mathrm{{ms}}$ ({analysis['h_crit_rk4']:.6f} s)",
        transform=ax2.transAxes,
        fontsize=7.9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#fff7bc", edgecolor="#d95f0e", alpha=0.93),
    )

    ax2.legend(
        loc="upper right",
        fontsize=7.8,
        framealpha=0.94,
    )

    fig.suptitle(
        "Análise Espectral, Discos de Gershgorin e Condicionamento — Conectoma da Drosophila (PB, N = 120)",
        fontsize=12.4,
        fontweight="bold",
        y=0.96,
    )

    saved_paths: List[Path] = []
    for out_file in target_paths:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300)
        saved_paths.append(out_file)

    if show:
        plt.show()
    plt.close(fig)
    return saved_paths
