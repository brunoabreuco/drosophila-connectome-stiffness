"""
Módulo de aquisição, limpeza e construção matricial do conectoma da Drosophila melanogaster.
Dataset: Janelia Hemibrain v1.2.1 (NeuPrint API) — Região: Protocerebral Bridge (PB).
Projeto Final de Cálculo Numérico — ICT / UNIFESP.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import certifi
    _HAS_CERTIFI = True
except ImportError:
    _HAS_CERTIFI = False


NEUPRINT_CUSTOM_URL = "https://neuprint.janelia.org/api/custom/custom"
DEFAULT_DATASET = "hemibrain:v1.2.1"
DEFAULT_ROI = "PB"
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/brunoabreuco/drosophila-connectome-stiffness/main/data"
)


def load_env_token(env_path: str | Path = ".env") -> Optional[str]:
    """
    Lê o token NEUPRINT_TOKEN de variável de ambiente, arquivo .env local
    ou google.colab.userdata (quando executado no Google Colab).
    Nunca expõe nem grava o token em arquivos rastreados pelo Git.
    """
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if token:
        return token

    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("NEUPRINT_TOKEN="):
                    val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val

    # Tentativa opcional via Google Colab Secrets
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get("NEUPRINT_TOKEN")
        if val:
            return val.strip()
    except Exception:
        pass

    return None


def _create_ssl_context() -> ssl.SSLContext:
    """Cria contexto SSL utilizando o bundle CA do certifi quando disponível."""
    if _HAS_CERTIFI:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def run_cypher_query(
    cypher: str,
    token: str,
    dataset: str = DEFAULT_DATASET,
    timeout: int = 45,
) -> Dict[str, Any]:
    """
    Executa uma consulta Cypher na API REST do NeuPrint com compressão gzip.
    """
    payload = json.dumps({"cypher": cypher, "dataset": dataset}).encode("utf-8")
    req = urllib.request.Request(
        NEUPRINT_CUSTOM_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
        method="POST",
    )
    ssl_ctx = _create_ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
        raw = resp.read()
        if resp.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def fetch_roi_subnetwork(
    token: str,
    roi: str = DEFAULT_ROI,
    n_main: int = 120,
    n_small: int = 20,
    dataset: str = DEFAULT_DATASET,
) -> Dict[str, Any]:
    """
    Extrai a sub-rede sináptica da região funcional `roi` (padrão: Protocerebral Bridge 'PB')
    do conectoma Hemibrain v1.2.1 em duas etapas otimizadas no servidor Neo4j:

    Etapa 1: Filtra neurônios com status = 'Traced' na ROI, remove autossinapses
             (n.bodyId <> m.bodyId) e ranqueia pelo grau sináptico total dentro da ROI.
    Etapa 2: Extrai todas as conexões direcionadas (pre)-[c:ConnectsTo]->(post) entre os
             Top N=120 neurônios e monta as matrizes W_N120 e W_N20 na convenção:
                 W_{ij} = peso sináptico de j (pré-sináptico) para i (pós-sináptico),
             com diagonal zerada (W_{ii} = 0).
    """
    start_time = time.perf_counter()

    # Etapa 1: Ranqueamento no servidor Neo4j de todos os neurônios Traced da ROI
    cypher_rank = f"""
    MATCH (n:Neuron {{`{roi}`: true}})-[c:ConnectsTo]-(m:Neuron {{`{roi}`: true}})
    WHERE n.status = "Traced" AND m.status = "Traced" AND n.bodyId <> m.bodyId
    RETURN n.bodyId AS bodyId,
           n.type AS type,
           n.instance AS instance,
           sum(c.weight) AS total_roi_weight,
           count(c) AS num_roi_edges
    ORDER BY total_roi_weight DESC, bodyId ASC
    """
    rank_response = run_cypher_query(cypher_rank, token=token, dataset=dataset)
    rank_rows = rank_response.get("data", [])
    total_traced_connected = len(rank_rows)
    total_roi_directed_edges = sum(int(row[4]) for row in rank_rows) // 2

    if total_traced_connected < n_main:
        raise RuntimeError(
            f"ROI '{roi}' has only {total_traced_connected} connected neurons (< {n_main})."
        )

    top_rows = rank_rows[:n_main]
    top_ids = [int(row[0]) for row in top_rows]

    # Etapa 2: Busca das conexões direcionadas no subgrafo induzido pelos Top N neurônios
    ids_csv = ", ".join(str(body_id) for body_id in top_ids)
    cypher_edges = f"""
    MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
    WHERE pre.bodyId IN [{ids_csv}]
      AND post.bodyId IN [{ids_csv}]
      AND pre.bodyId <> post.bodyId
    RETURN pre.bodyId AS pre_id,
           post.bodyId AS post_id,
           c.weight AS weight
    """
    edges_response = run_cypher_query(cypher_edges, token=token, dataset=dataset)
    edge_rows = edges_response.get("data", [])

    id_to_idx = {body_id: idx for idx, body_id in enumerate(top_ids)}
    W_main = np.zeros((n_main, n_main), dtype=np.float64)

    for pre_id, post_id, weight in edge_rows:
        pre_body_id = int(pre_id)
        post_body_id = int(post_id)
        if pre_body_id == post_body_id:
            continue
        col_j = id_to_idx[pre_body_id]   # Coluna j: neurônio pré-sináptico (origem)
        row_i = id_to_idx[post_body_id]  # Linha i: neurônio pós-sináptico (destino)
        W_main[row_i, col_j] += float(weight)

    # Garantia explícita de ausência de autossinapses (W_ii = 0)
    np.fill_diagonal(W_main, 0.0)

    # Subsistema reduzido N=20 (Top 20 neurônios do mesmo ranking)
    W_small = W_main[:n_small, :n_small].copy()
    np.fill_diagonal(W_small, 0.0)

    d_in_main = np.sum(W_main, axis=1)
    d_out_main = np.sum(W_main, axis=0)
    total_sub_main = d_in_main + d_out_main

    # Verificação de conectividade de todos os neurônios no subgrafo induzido N=120
    if np.any(total_sub_main <= 0):
        disconnected = np.where(total_sub_main <= 0)[0]
        raise RuntimeError(f"Disconnected neurons in subgraph N={n_main}: {disconnected}")

    neurons: List[Dict[str, Any]] = []
    for idx, row in enumerate(top_rows):
        body_id = int(row[0])
        neuron_type = str(row[1]) if row[1] is not None else "Unknown"
        neuron_instance = str(row[2]) if row[2] is not None else "Unknown"
        roi_weight = int(row[3])
        neurons.append(
            {
                "index": idx,
                "bodyId": body_id,
                "type": neuron_type,
                "instance": neuron_instance,
                "d_in": int(round(d_in_main[idx])),
                "d_out": int(round(d_out_main[idx])),
                "total_weight": int(round(total_sub_main[idx])),
                "total_roi_weight": roi_weight,
            }
        )

    elapsed_seconds = time.perf_counter() - start_time
    return {
        "W_N120": W_main,
        "W_N20": W_small,
        "neurons_N120": neurons,
        "roi": roi,
        "dataset": dataset,
        "total_traced_connected_in_roi": total_traced_connected,
        "total_directed_edges_in_roi": total_roi_directed_edges,
        "elapsed_seconds": elapsed_seconds,
    }


def save_subnetwork_csv(
    data: Dict[str, Any],
    data_dir: str | Path = "data",
) -> Tuple[Path, Path, Path]:
    """
    Salva as matrizes W_real_N120.csv, W_real_N20.csv e a tabela neurons_N120.csv em `data/`.
    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_w120 = out_dir / "W_real_N120.csv"
    path_w20 = out_dir / "W_real_N20.csv"
    path_neurons = out_dir / "neurons_N120.csv"

    np.savetxt(path_w120, data["W_N120"], delimiter=",", fmt="%.0f")
    np.savetxt(path_w20, data["W_N20"], delimiter=",", fmt="%.0f")

    fieldnames = [
        "index",
        "bodyId",
        "type",
        "instance",
        "d_in",
        "d_out",
        "total_weight",
        "total_roi_weight",
    ]
    with open(path_neurons, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in data["neurons_N120"]:
            writer.writerow(row)

    return path_w120, path_w20, path_neurons


def load_connectome_data(
    data_dir: str | Path = "data",
    github_raw_base: str = GITHUB_RAW_BASE,
    token: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Carrega W_real_N120.csv, W_real_N20.csv e neurons_N120.csv do diretório local `data/`.
    Caso os arquivos locais não existam, tenta extrair via API NeuPrint (se `token` for fornecido)
    ou realiza o download automático a partir do repositório GitHub.
    """
    out_dir = Path(data_dir)
    path_w120 = out_dir / "W_real_N120.csv"
    path_w20 = out_dir / "W_real_N20.csv"
    path_neurons = out_dir / "neurons_N120.csv"

    if not (path_w120.exists() and path_w20.exists() and path_neurons.exists()):
        if token:
            data = fetch_roi_subnetwork(token=token)
            save_subnetwork_csv(data, data_dir=out_dir)
            return data["W_N120"], data["W_N20"], data["neurons_N120"]

        out_dir.mkdir(parents=True, exist_ok=True)
        ssl_ctx = _create_ssl_context()
        for filename in ["W_real_N120.csv", "W_real_N20.csv", "neurons_N120.csv"]:
            target = out_dir / filename
            if not target.exists():
                url = f"{github_raw_base}/{filename}"
                req = urllib.request.Request(url, headers={"User-Agent": "UNIFESP-CN/1.0"})
                with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                    target.write_bytes(resp.read())

    W_n120 = np.loadtxt(path_w120, delimiter=",", dtype=np.float64)
    W_n20 = np.loadtxt(path_w20, delimiter=",", dtype=np.float64)

    neurons: List[Dict[str, Any]] = []
    with open(path_neurons, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            neurons.append(
                {
                    "index": int(row["index"]),
                    "bodyId": int(row["bodyId"]),
                    "type": row["type"],
                    "instance": row["instance"],
                    "d_in": int(float(row["d_in"])),
                    "d_out": int(float(row["d_out"])),
                    "total_weight": int(float(row["total_weight"])),
                    "total_roi_weight": int(
                        float(row.get("total_roi_weight", row["total_weight"]))
                    ),
                }
            )

    return W_n120, W_n20, neurons


def build_system_matrix(
    W: np.ndarray,
    epsilon: float = 1.0,
    mode: str = "laplacian",
) -> np.ndarray:
    """
    Constrói a matriz de estado A do sistema linear de EDOs dx/dt = A x.

    Parâmetros:
    -----------
    W : np.ndarray (N x N)
        Matriz de conectividade sináptica onde W_{ij} é o peso de j -> i (com W_{ii} = 0).
    epsilon : float (padrão = 1.0 s^-1)
        Taxa basal de decaimento/vazamento passivo de membrana.
    mode : str
        - 'laplacian': A = -epsilon * I - D_in + W, onde D_in = diag(d_1, ..., d_N)
                       e d_i = sum_j W_{ij} (condutância sináptica de entrada).
        - 'shift':     A = -gamma * I + W, onde gamma = max_k Re(lambda_k(W)) + epsilon.
    """
    W_clean = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(W_clean, 0.0)
    n = W_clean.shape[0]
    identity = np.eye(n, dtype=np.float64)

    if mode == "laplacian":
        d_in = np.sum(W_clean, axis=1)
        return -epsilon * identity - np.diag(d_in) + W_clean
    elif mode == "shift":
        eigenvalues_W = np.linalg.eigvals(W_clean)
        gamma = float(np.max(np.real(eigenvalues_W))) + epsilon
        return -gamma * identity + W_clean
    else:
        raise ValueError("Invalid mode. Choose 'laplacian' or 'shift'.")


def compute_spectral_metrics(W: np.ndarray, A: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas estruturais e espectrais da matriz de conectividade W e da matriz A.
    """
    n = W.shape[0]
    nnz = int(np.count_nonzero(W))
    total_entries = n * n
    sparsity_pct = 100.0 * (1.0 - nnz / total_entries)
    density_pct = 100.0 * (nnz / total_entries)

    d_in = np.sum(W, axis=1)
    d_out = np.sum(W, axis=0)
    active_weights = W[W > 0]

    eigenvalues = np.linalg.eigvals(A)
    real_abs = np.abs(np.real(eigenvalues))
    mod_abs = np.abs(eigenvalues)

    return {
        "N": float(n),
        "nnz": float(nnz),
        "total_entries": float(total_entries),
        "sparsity_pct": float(sparsity_pct),
        "density_pct": float(density_pct),
        "total_synapses": float(np.sum(W)),
        "d_in_min": float(np.min(d_in)),
        "d_in_max": float(np.max(d_in)),
        "d_in_mean": float(np.mean(d_in)),
        "d_in_median": float(np.median(d_in)),
        "d_in_std": float(np.std(d_in)),
        "d_out_min": float(np.min(d_out)),
        "d_out_max": float(np.max(d_out)),
        "w_max": float(np.max(W)),
        "w_mean_active": float(np.mean(active_weights)) if nnz > 0 else 0.0,
        "w_median_active": float(np.median(active_weights)) if nnz > 0 else 0.0,
        "lambda_re_min_abs": float(np.min(real_abs)),
        "lambda_re_max_abs": float(np.max(real_abs)),
        "lambda_mod_max": float(np.max(mod_abs)),
        "stiffness_ratio_S": float(np.max(real_abs) / np.min(real_abs)),
        "cond_2_A": float(np.linalg.cond(A, p=2)),
        "h_crit_euler": float(2.0 / np.max(mod_abs)),
        "h_crit_rk4": float(2.785293563405289 / np.max(mod_abs)),
    }


def _neuron_class_color(neuron_type: str) -> str:
    """Retorna cor semântica para as principais famílias celulares da Protocerebral Bridge."""
    if neuron_type == "LPsP":
        return "#d62728"  # Vermelho intenso (Mega-hubs LPsP)
    if neuron_type.startswith("Delta7"):
        return "#ff7f0e"  # Laranja escuro (Hubs inibitórios locais Delta7)
    if neuron_type.startswith("EPG"):
        return "#1f77b4"  # Azul clássico (Neurônios de bússola EPG)
    if neuron_type.startswith("PEN"):
        return "#2ca02c"  # Verde (Neurônios de velocidade angular PEN_a / PEN_b)
    return "#7f7f7f"      # Cinza (Neurônios modulatórios/extrínsecos: ExR3, OA-VPM3, etc.)


def plot_connectome_hubs(
    W: np.ndarray,
    neurons: List[Dict[str, Any]],
    save_paths: Optional[Sequence[str | Path] | str | Path] = None,
    show: bool = False,
) -> List[Path]:
    """
    Gera a Figura 1 (300 DPI, qualidade de publicação em português):
      - Painel (a): Mapa de calor de log10(1 + W_ij) para N = 120 neurônios da Protocerebral Bridge (PB).
      - Painel (b): Distribuição e perfil de grau de entrada ponderado d_i = sum_j W_ij,
                    destacando a disparidade entre neurônios periféricos (d_min) e hubs (d_max).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if save_paths is None:
        target_paths: List[Path] = [
            Path("results/figures/fig1_connectome_hubs.png"),
            Path("report/fig1_connectome_hubs.png"),
        ]
    elif isinstance(save_paths, (str, Path)):
        target_paths = [Path(save_paths)]
    else:
        target_paths = [Path(p) for p in save_paths]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.0, 6.2), gridspec_kw={"width_ratios": [1.05, 1.18]}
    )
    fig.subplots_adjust(wspace=0.27, top=0.81, bottom=0.13, left=0.06, right=0.97)

    # -------------------------------------------------------------------------
    # Painel (a): Mapa de calor da matriz de pesos sinápticos W em escala log10(1 + W_ij)
    # -------------------------------------------------------------------------
    W_log = np.log10(1.0 + W)
    im = ax1.imshow(W_log, cmap="magma", aspect="equal", interpolation="nearest", origin="upper")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.035)
    cbar.set_label(r"Intensidade sináptica $\log_{10}(1 + W_{ij})$", fontsize=10)

    # Linhas tracejadas delimitando o bloco dos hubs principais (LPsP + Delta7: índices 0 a 43)
    hub_boundary = 43.5
    ax1.axhline(hub_boundary, color="#00e5ff", linestyle="--", linewidth=1.1, alpha=0.85)
    ax1.axvline(hub_boundary, color="#00e5ff", linestyle="--", linewidth=1.1, alpha=0.85)

    ax1.text(
        21.5,
        -3.5,
        "Hubs Globais\n(LPsP + Δ7)",
        color="#b33900",
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    ax1.text(
        81.5,
        -3.5,
        "Neurônios Colunares / Anelares\n(EPG, PEN_a, PEN_b, outros)",
        color="#1f4e78",
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )

    nnz = int(np.count_nonzero(W))
    sparsity = 100.0 * (1.0 - nnz / (W.shape[0] ** 2))
    ax1.set_title(
        "(a) Matriz de Conectividade Sináptica $W$ ($N = 120$)",
        fontweight="bold",
        pad=32,
    )
    ax1.set_xlabel("Índice do neurônio pré-sináptico $j$ (origem $j \\to i$)")
    ax1.set_ylabel("Índice do neurônio pós-sináptico $i$ (destino $j \\to i$)")
    ax1.set_xticks([0, 20, 43, 60, 80, 100, 119])
    ax1.set_yticks([0, 20, 43, 60, 80, 100, 119])

    ax1.text(
        0.03,
        0.04,
        f"Conexões ativas: {nnz:,} / 14.400\nEsparsidade: {sparsity:.1f}% ($W_{{ii}} = 0$)".replace(",", "."),
        transform=ax1.transAxes,
        fontsize=8.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.72, edgecolor="#00e5ff"),
    )

    # -------------------------------------------------------------------------
    # Painel (b): Perfil de grau de entrada d_i ordenado + Inset de histograma
    # -------------------------------------------------------------------------
    d_in = np.sum(W, axis=1)
    order_din = np.argsort(-d_in)
    d_in_sorted = d_in[order_din]
    ranks = np.arange(1, len(d_in) + 1)

    colors_sorted = [
        _neuron_class_color(str(neurons[idx]["type"])) for idx in order_din
    ]

    ax2.bar(ranks, d_in_sorted, color=colors_sorted, width=0.88, edgecolor="none", alpha=0.92)
    mean_din = float(np.mean(d_in))
    ax2.axhline(
        mean_din,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
    )
    ax2.text(
        8.0,
        mean_din + 240.0,
        f"Média $\\bar{{d}}_{{\\mathrm{{in}}}} = {mean_din:.1f}$ sinapses",
        fontsize=8.2,
        fontweight="bold",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
    )

    # Anotações de d_max (Hub) e d_min (Periférico)
    idx_max = int(order_din[0])
    idx_min = int(order_din[-1])
    hub_info = neurons[idx_max]
    periph_info = neurons[idx_min]
    d_max = float(d_in[idx_max])
    d_min = float(d_in[idx_min])

    ax2.annotate(
        f"Maior Hub ($d_{{\\max}} = {int(d_max)}$)\n"
        f"{hub_info['type']} (ID {hub_info['bodyId']})",
        xy=(1, d_max),
        xytext=(5, d_max * 0.44),
        fontsize=8.4,
        fontweight="bold",
        color="#7f2704",
        arrowprops=dict(arrowstyle="->", color="#7f2704", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5eb", edgecolor="#fd8d3c"),
    )

    ax2.annotate(
        f"Periférico ($d_{{\\min}} = {int(d_min)}$)\n"
        f"{periph_info['type']} (ID {periph_info['bodyId']})\n"
        f"Disparidade $d_{{\\max}}/d_{{\\min}} = {d_max/d_min:.1f}\\times$",
        xy=(120, d_min),
        xytext=(73, d_max * 0.34),
        fontsize=8.4,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7", edgecolor="#969696"),
    )

    ax2.set_title(
        "(b) Condutância Sináptica de Entrada $d_i = \\sum_j W_{ij}$ e Hubs",
        fontweight="bold",
        pad=32,
    )
    ax2.set_xlabel("Posto (rank) do neurônio ordenado por $d_i$ decrescente")
    ax2.set_ylabel("Grau ponderado de entrada $d_i$ (nº de sinapses)")
    ax2.set_xlim(0, 122)
    ax2.set_ylim(0, d_max * 1.13)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Legenda das classes celulares
    legend_elements = [
        Patch(facecolor="#d62728", label="Mega-hubs LPsP (2)"),
        Patch(facecolor="#ff7f0e", label="Hubs locais $\\Delta 7$ (42)"),
        Patch(facecolor="#1f77b4", label="Bússola EPG (40)"),
        Patch(facecolor="#2ca02c", label="Rotação PEN_a / PEN_b (31)"),
        Patch(facecolor="#7f7f7f", label="Modulatórios / outros (5)"),
    ]
    ax2.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7.9,
        framealpha=0.94,
        title="Família Celular (PB)",
        title_fontsize=8.2,
    )

    # Inset: Histograma da distribuição de d_i posicionado sem sobrepor a legenda
    ax_inset = ax2.inset_axes([0.34, 0.54, 0.26, 0.33])
    bins = np.linspace(0, d_max * 1.02, 16)
    ax_inset.hist(d_in, bins=bins, color="#3182bd", edgecolor="white", alpha=0.88)
    ax_inset.axvline(mean_din, color="#222222", linestyle="--", linewidth=1.0)
    ax_inset.axvline(d_max, color="#d62728", linestyle="-", linewidth=1.1)
    ax_inset.set_title("Histograma de $d_i$", fontsize=7.9, fontweight="bold", pad=4)
    ax_inset.set_xlabel("$d_i$ (sinapses)", fontsize=7.3)
    ax_inset.set_ylabel("Neurônios", fontsize=7.3)
    ax_inset.tick_params(labelsize=6.8)
    ax_inset.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "Conectoma Real da Drosophila melanogaster (Hemibrain v1.2.1 — Protocerebral Bridge, N = 120)",
        fontsize=12.5,
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
