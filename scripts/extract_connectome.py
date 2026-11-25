"""
Script da Parte 1 — Aquisição e Limpeza do Conectoma (NeuPrint API) + Construção da Matriz A e Figura 1.
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

from src.connectome import (
    build_system_matrix,
    compute_spectral_metrics,
    fetch_roi_subnetwork,
    load_connectome_data,
    load_env_token,
    plot_connectome_hubs,
    save_subnetwork_csv,
)


def main() -> None:
    print("=" * 78)
    print("PARTE 1 — AQUISIÇÃO DO CONECTOMA (NEUPRINT) E CONSTRUÇÃO DA MATRIZ A")
    print("=" * 78)

    token = load_env_token(ROOT_DIR / ".env")
    if token:
        print("[1/4] Token NEUPRINT_TOKEN carregado com segurança a partir de .env.")
        print("[2/4] Consultando API REST NeuPrint (hemibrain:v1.2.1, ROI='PB')...")
        subnetwork_data = fetch_roi_subnetwork(token=token, roi="PB", n_main=120, n_small=20)
        W_n120 = subnetwork_data["W_N120"]
        W_n20 = subnetwork_data["W_N20"]
        neurons = subnetwork_data["neurons_N120"]
        print(
            f"      -> {subnetwork_data['total_traced_connected_in_roi']} neurônios 'Traced' conectados na PB "
            f"({subnetwork_data['total_directed_edges_in_roi']} conexões direcionadas) extraídos em "
            f"{subnetwork_data['elapsed_seconds']:.2f} s."
        )
        path_w120, path_w20, path_neurons = save_subnetwork_csv(
            subnetwork_data, data_dir=ROOT_DIR / "data"
        )
        print(f"[3/4] Arquivos salvos em:\n      - {path_w120}\n      - {path_w20}\n      - {path_neurons}")
    else:
        print("[1/4] Token não encontrado no ambiente; carregando matrizes de data/...")
        W_n120, W_n20, neurons = load_connectome_data(data_dir=ROOT_DIR / "data")

    # Construção das matrizes A (N=120 e N=20) com epsilon = 1.0
    A_n120 = build_system_matrix(W_n120, epsilon=1.0, mode="laplacian")
    A_n20 = build_system_matrix(W_n20, epsilon=1.0, mode="laplacian")
    assert A_n120.shape == (120, 120) and A_n20.shape == (20, 20)

    metrics_120 = compute_spectral_metrics(W_n120, A_n120)
    metrics_20 = compute_spectral_metrics(W_n20, A_n20)

    figure_paths = plot_connectome_hubs(
        W_n120,
        neurons,
        save_paths=[
            ROOT_DIR / "results" / "figures" / "fig1_connectome_hubs.png",
            ROOT_DIR / "report" / "fig1_connectome_hubs.png",
        ],
        show=False,
    )
    print("[4/4] Figura 1 gerada com sucesso (300 DPI):")
    for fig_path in figure_paths:
        print(f"      - {fig_path}")

    print("\n--- RESUMO TÉCNICO: SUB-REDE PRINCIPAL (N = 120) ---")
    for metric_name, metric_val in metrics_120.items():
        print(f"  {metric_name:20s}: {metric_val:.4f}")

    print("\n--- RESUMO TÉCNICO: SUBSISTEMA DE VALIDAÇÃO (N = 20) ---")
    for metric_name, metric_val in metrics_20.items():
        print(f"  {metric_name:20s}: {metric_val:.4f}")


if __name__ == "__main__":
    main()
