"""
phi_demais_hiper.py
===================
Analyses default hyperparameters (phi_b and RF parameters) for all nine
detectors ranked by mean macro F1-score.

Two scenarios: C1 (drift) and C3 (catch24_drift).

Produces per scenario:
  1. phi_b strip plot (all 9 detectors) with bias indicator per point.
  2. RF hyperparameter panel (best vs worst detector).

Bias indicator (marker edge colour):
  Red   → memory > min class segment  (potential bias)
  Grey  → memory ≤ min class segment  (no bias)
  White → dataset files not found     (unknown)
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURAÇÕES
# ============================================================

RESULT_ROOT   = Path("exp_otimizacao/result_reais_completo")
SUMMARY_DIR   = RESULT_ROOT / "resumo_tabelas"
MEAN_CSV      = SUMMARY_DIR / "tabela_media_f1.csv"
OUTPUT_DIR    = SUMMARY_DIR / "figuras_artigo_delta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_ROOT = Path("exp_otimizacao/result_reais_completo/datasets")

# Fonte dos resultados: a MESMA do teste_reais_agr_datasets.py, que gera as
# Tabelas S-11 e S-15, os testes estatisticos e a Figura 7, isto e, a pasta
# original <dataset>/<cenario>/<detector>/. A pasta de rerun
#   <dataset>/resultados_catch24_drift_rerun/<cenario>/<detector>/
# so e usada se o arquivo pedido nao existir na pasta original (com aviso).
# Antes o rerun tinha prioridade, e o MH (HDDMA) saia com F1 diferente da
# Tabela S-15 (C1 75,94 em vez de 82,30 e C3 93,18 em vez de 96,38).
# O F1 e lido de df_f1_all.csv (coluna RF, media dos folds, em %).
RERUN_DIR     = "resultados_catch24_drift_rerun"
COLUNA_F1     = "RF"
DECAY_THRESHOLD = 0.01   # S_j < 1% → memory exhausted

SCENARIOS = {"C1": "drift", "C3": "catch24_drift"}

ABBREV = {
    "gait"             : "Gait",
    #"idosos"           : "HAR70+",
    "mhealth"          : "MH",
    "occ"              : "RO",
    "pamap2_ankle"     : "P2-Ankle",
    "pamap2_chest"     : "P2-Chest",
    "pamap2_hand"      : "P2-Hand",
    "rs"               : "DR",
    "smartphone"       : "SmPh",
    "usc_had"          : "USC",
    "ward_left_ankle"  : "W-LAnkle",
    "ward_left_wrist"  : "W-LWrist",
    "ward_right_ankle" : "W-RAnkle",
    "ward_right_wrist" : "W-RWrist",
    "ward_waist"       : "W-Waist",
    "sp_har"            : "SP-HAR",
    "sw_har"            : "SW-HAR",
}

ALL_DETECTORS = [
    "ADWIN", "PageHinkley", "KSWIN", "CUSUM",
    "EWMAChart", "GeometricMovingAverage",
    "HDDMAverage", "HDDMWeighted", "SEED",
]

DET_SHORT = {
    "ADWIN"                 : "ADWIN",
    "PageHinkley"           : "PH",
    "KSWIN"                 : "KSWIN",
    "CUSUM"                 : "CUSUM",
    "EWMAChart"             : "EWMAC",
    "GeometricMovingAverage": "GMA",
    "HDDMAverage"           : "HDDMA",
    "HDDMWeighted"          : "HDDMW",
    "SEED"                  : "SEED",
}

# 9 colours + markers — colour-blind-friendly palette
DET_STYLE = {
    "ADWIN"                 : ("#0C447C", "o"),   # dark blue
    "PageHinkley"           : ("#E69F00", "s"),   # orange
    "KSWIN"                 : ("#009E73", "^"),   # green
    "CUSUM"                 : ("#CC79A7", "D"),   # pink
    "EWMAChart"             : ("#56B4E9", "v"),   # light blue
    "GeometricMovingAverage": ("#D55E00", "P"),   # red-orange
    "HDDMAverage"           : ("#F0E442", "*"),   # yellow
    "HDDMWeighted"          : ("#6A3D9A", "X"),   # purple
    "SEED"                  : ("#333333", "h"),   # dark grey
}

# Bias edge colours — 3 levels
EDGE_NONE     = "#555555"   # dark grey → within class  (memory < mean class segment)
EDGE_POSSIBLE = "#E69F00"   # orange    → between classes (mean class ≤ memory < mean block)
EDGE_BIAS     = "#CC0000"   # red       → possible bias  (memory ≥ mean block size)
EDGE_UNKNOWN  = "white"     # white     → unknown

RF_PARAMS = [
    ("rf_n_estimators",      "n_estimators",      "int",        None),
    ("rf_max_depth",         "max_depth",          "int",        None),
    ("rf_min_samples_split", "min_samples_split",  "int",        None),
    ("rf_min_samples_leaf",  "min_samples_leaf",   "int",        None),
    ("rf_max_features",      "max_features",       "categorical",["sqrt", "log2"]),
    ("rf_bootstrap",         "bootstrap",          "boolean",    ["True", "False"]),
]

# ============================================================
# DATASET FILE LAYOUTS
# ============================================================

DATASET_LAYOUTS = {
    "mhealth" : ("dados_mhealth", "Activity",              "subject_subject", ".csv",
                 [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "rs"      : ("rs",            "rotulo",                "base",            "_final.csv",
                 [[3],[4],[5],[0,1],[0,2],[0,7],[0,8],[6,1],[6,2],[6,7],[6,8]]),
    "gait"    : ("gait",          "label",                 "base",            ".csv",
                 [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "idosos"  : ("idosos",        "label",                 "base",            ".csv",
                 [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14]]),
    "pamap2_ankle" : ("pamap2",   "activityID",            "base_",           ".csv",
                      [[0],[1],[2],[3],[4],[5],[6],[7]]),
    "pamap2_chest" : ("pamap2",   "activityID",            "base_",           ".csv",
                      [[0],[1],[2],[3],[4],[5],[6],[7]]),
    "pamap2_hand"  : ("pamap2",   "activityID",            "base_",           ".csv",
                      [[0],[1],[2],[3],[4],[5],[6],[7]]),
    "occ"     : ("occ",           "Room_Occupancy_Count",  "occupancy_dia_",  ".csv",
                 [[0,2],[0,3],[0,4],[0,5],[0,6],[1,2],[1,3],[1,4],[1,5],[1,6]]),
    "smartphone": ("smartphone",  "label",                 "user_",           ".csv",
                 [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],
                  [15,16,17],[18,19,20],[21,22,23],[24,25,26],[27,28,29]]),
    "usc_had" : ("usc_had",       "activity_code",         "usc_had_subject_",".csv",
                 [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]),
    "ward_left_wrist" : ("WARD",  "label", "base_", ".csv",
                         [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "ward_right_wrist": ("WARD",  "label", "base_", ".csv",
                         [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "ward_waist"      : ("WARD",  "label", "base_", ".csv",
                         [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "ward_left_ankle" : ("WARD",  "label", "base_", ".csv",
                         [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "ward_right_ankle": ("WARD",  "label", "base_", ".csv",
                         [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]),
    "sw_har": ("sw_har",  "rotulo", "sujeito_", ".csv",
                         [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21,22]]),
    "sp_har": ("sp_har",  "rotulo", "sujeito_", ".csv",
                         [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21,22]]),
}

# ============================================================
# BIAS ASSESSMENT
# ============================================================

def class_segment_stats(dataset_key: str) -> dict | None:
    """
    Returns a dict with:
        min_seg    : minimum consecutive samples of any class in any block
        median_seg : median of all class-run lengths across all blocks
        max_seg    : maximum consecutive samples of any class in any block
        block_size : mean total samples per block (fold)
    Returns None if files are not found.
    """
    if dataset_key not in DATASET_LAYOUTS:
        return None
    subfolder, label_col, nome_base, end_base, folds = DATASET_LAYOUTS[dataset_key]
    data_dir = DATASETS_ROOT / subfolder
    all_indices = sorted({idx for fold in folds for idx in fold})

    all_runs   = []   # all consecutive run lengths
    block_sizes = []  # total samples per file
    found_any = False

    for idx in all_indices:
        fpath = data_dir / f"{nome_base}{idx}{end_base}"
        if not fpath.exists():
            continue
        found_any = True
        try:
            df = pd.read_csv(fpath, usecols=[label_col])
        except Exception:
            continue
        labels = df[label_col].values
        if len(labels) == 0:
            continue
        block_sizes.append(len(labels))
        run_len = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run_len += 1
            else:
                all_runs.append(run_len)
                run_len = 1
        all_runs.append(run_len)   # last run

    if not found_any:
        print(f"  [WARN] No files found for '{dataset_key}' in {data_dir}")
        return None
    if not all_runs:
        return None

    return {
        "min_seg"    : int(np.min(all_runs)),
        "mean_seg"   : float(np.mean(all_runs)),
        "median_seg" : float(np.median(all_runs)),
        "max_seg"    : int(np.max(all_runs)),
        "block_size" : float(np.mean(block_sizes)),
    }


def memory_instances(phi_b: float, threshold: float = DECAY_THRESHOLD) -> int:
    S, w = 1.0, 0
    for i in range(1, 5_000_000):
        w += 1
        S = (1 - phi_b * np.log(1 + w)) * S
        if S < threshold:
            return i
    return 5_000_000


def bias_level(phi_b, stats: dict | None) -> str:
    """
    Classifies bias into 3 levels based on memory vs class/block statistics.

    Levels:
      EDGE_NONE     : memory < mean class segment  → within class
      EDGE_POSSIBLE : mean class ≤ memory < mean block size → between classes
      EDGE_BIAS     : memory ≥ mean block size → possible bias
      EDGE_UNKNOWN  : phi_b or stats missing
    """
    if phi_b is None or (isinstance(phi_b, float) and np.isnan(phi_b)):
        return EDGE_UNKNOWN
    if stats is None:
        return EDGE_UNKNOWN
    mem = memory_instances(phi_b)
    if mem < stats["mean_seg"]:
        return EDGE_NONE
    elif mem < stats["block_size"]:
        return EDGE_POSSIBLE
    else:
        return EDGE_BIAS


# Pre-compute segment stats once
print("[INFO] Computing class segment statistics per dataset...")
DATASET_STATS = {}
for ds in ABBREV:
    stats = class_segment_stats(ds)
    DATASET_STATS[ds] = stats
    if stats:
        print(f"  {ABBREV[ds]:12s}  min={stats['min_seg']:,}  "
              f"mean={stats['mean_seg']:,.0f}  "
              f"median={stats['median_seg']:,.0f}  "
              f"max={stats['max_seg']:,}  "
              f"block={stats['block_size']:,.0f}")
    else:
        print(f"  {ABBREV[ds]:12s}  N/A")

# ============================================================
# CSV / JSON LOADING
# ============================================================

def pasta_resultado(ds: str, cenario: str, detector: str | None,
                    arquivos=("df_f1_all.csv",)) -> Path:
    """Pasta do resultado na mesma fonte da Tabela S-15 (pasta original).
    O rerun so e usado se nenhum dos arquivos pedidos existir na original."""
    if detector is None:
        return RESULT_ROOT / ds / cenario
    orig = RESULT_ROOT / ds / cenario / detector
    if any((orig / a).exists() for a in arquivos):
        return orig
    rr = RESULT_ROOT / ds / RERUN_DIR / cenario / detector
    if any((rr / a).exists() for a in arquivos):
        print(f"  [AVISO] {ds}/{cenario}/{detector}: {arquivos} ausente na pasta "
              f"original, usando o rerun")
        return rr
    return orig


def f1_medio(pasta: Path) -> float:
    """Macro F1 (%) medio dos folds, como na Tabela S-15."""
    arq = pasta / "df_f1_all.csv"
    if not arq.exists():
        return np.nan
    df = pd.read_csv(arq)
    if not {"fold", COLUNA_F1}.issubset(df.columns):
        print(f"  [WARN] colunas inesperadas em {arq}")
        return np.nan
    return float(100 * df.groupby("fold")[COLUNA_F1].mean().mean())


def build_mean_df() -> pd.DataFrame:
    linhas = {}
    for ds in ABBREV:
        if not (RESULT_ROOT / ds).exists():
            print(f"  [MISS] dataset {ds}")
            continue
        r = {"baseline": f1_medio(pasta_resultado(ds, "baseline", None)),
             "catch24":  f1_medio(pasta_resultado(ds, "catch24", None))}
        for det in ALL_DETECTORS:
            for cen in ("drift", "catch24_drift"):
                pasta = pasta_resultado(ds, cen, det)
                r[f"{cen}_{det}"] = f1_medio(pasta)

        linhas[ds] = r
    df = pd.DataFrame.from_dict(linhas, orient="index")
    df.index.name = "base"
    return df

mean_df  = build_mean_df()
datasets = mean_df.index.tolist()
mean_df.to_csv(OUTPUT_DIR / "f1_medio_por_dataset_usado_na_figura.csv")

def classify_col(col):
    col = col.strip()
    if col == "baseline": return "C0", None, False
    if col == "catch24":  return "C2", None, False
    m = re.fullmatch(r"catch24_drift_ot_(.+)", col)
    if m: return "C3", m.group(1), True
    m = re.fullmatch(r"catch24_drift_(.+)", col)
    if m: return "C3", m.group(1), False
    m = re.fullmatch(r"drift_ot_(.+)", col)
    if m: return "C1", m.group(1), True
    m = re.fullmatch(r"drift_(.+)", col)
    if m: return "C1", m.group(1), False
    return "OTHER", col, False

col_meta = {col: classify_col(col) for col in mean_df.columns}


def rank_detectors(scenario_label):
    det_mean = {}
    for col, (cat, det, opt) in col_meta.items():
        if cat == scenario_label and not opt and det in ALL_DETECTORS:
            vals = mean_df[col].dropna()
            if not vals.empty:
                det_mean[det] = vals.mean()
    if not det_mean:
        raise RuntimeError(f"No default columns for {scenario_label}.")
    return sorted(det_mean.items(), key=lambda x: x[1], reverse=True)


def _normalise_json(d):
    params = dict(d.get("best_params", {}))
    f1 = d.get("best_macro_f1") or d.get("best_value_mean_f1_macro", np.nan)
    if "scaler_model" not in params:
        params["scaler_model"] = np.nan
    return {"best_macro_f1": f1, **params}


def load_params(datasets, scenario_folder, detector):
    rows = []
    for ds in datasets:
        # mesma pasta do F1 (df_f1_all.csv), para phi_b e F1 virem do mesmo run
        base = pasta_resultado(ds, scenario_folder, detector)
        p = base / "resultado_otimizacao.json"
        if not p.exists():
            p = base / "resultado_otimizacao_rf.json"
        if not p.exists():
            print(f"  [MISS] {ds}/{scenario_folder}/{detector}: sem json de "
                  f"otimizacao em {base}")
            rows.append({"base": ds})
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append({"base": ds, **_normalise_json(d)})
    return pd.DataFrame(rows).set_index("base")

# ============================================================
# HELPERS
# ============================================================

def _get(df, col, base):
    if base in df.index and col in df.columns:
        v = df.at[base, col]
        return v if pd.notna(v) else np.nan
    return np.nan

def _phi_b(df, base):
    v = _get(df, "phi_b", base)
    return float(v) if pd.notna(v) else np.nan

# ============================================================
# FIGURE 1 — phi_b strip plot (all detectors)
# ============================================================

def phi_b_strip_plot_all(dfs: dict, scenario_label: str, ranked: list,
                          out_dir: Path):
    """
    dfs      : {detector_name: DataFrame}
    ranked   : ordered list of (detector_name, mean_f1) — top to bottom in legend
    """
    all_bases = sorted(
        set(b for df in dfs.values() for b in df.index),
        key=lambda x: ABBREV.get(x, x)
    )
    labels = [ABBREV.get(b, b) for b in all_bases]
    n = len(labels)
    n_det = len(ALL_DETECTORS)

    # vertical offsets: spread detectors evenly within ±0.40
    offsets = np.linspace(0.38, -0.38, n_det)

    fig, ax = plt.subplots(figsize=(9.0, max(5, n * 0.65 + 2.0)))

    # Row backgrounds
    for i in range(n):
        ax.axhspan(i - 0.5, i + 0.5,
                   color="#F4F4F2" if i % 2 == 0 else "white", zorder=0)

    # Log-decade guide lines
    for exp in range(-7, 1):
        ax.axvline(10**exp, color="#CCCCCC", lw=0.6, ls="--", zorder=1)

    y = np.arange(n)

    for det_idx, det in enumerate(ALL_DETECTORS):
        color, marker = DET_STYLE[det]
        df = dfs.get(det)
        off = offsets[det_idx]

        for i, b in enumerate(all_bases):
            pv = _phi_b(df, b) if df is not None else np.nan
            ec = bias_level(pv, DATASET_STATS.get(b))
            ax.scatter(
                pv if pd.notna(pv) else np.nan,
                y[i] + off,
                color=color, s=55, zorder=3,
                edgecolors=ec, linewidths=1.4,
                marker=marker
            )

    ax.set_xscale("log")
    ax.set_xlim(5e-8, 2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\phi_b$ (log scale)", fontsize=10)
    ax.set_title(rf"$\phi_b$ per dataset — {scenario_label} (default)",
                 fontsize=11, pad=10)
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.grid(axis="x", which="minor", color="#EEEEEE", lw=0.4, ls=":")

    # Legend — detectors ordered by F1 rank
    det_handles = []
    for det, mean_f1 in ranked:
        color, marker = DET_STYLE[det]
        det_handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=color,
                   markeredgecolor="white", markersize=7,
                   label=f"{DET_SHORT[det]} ({mean_f1:.1f})")
        )

    # Bias legend
    bias_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_NONE,     markeredgewidth=1.4, markersize=7,
               label="Mem. within class (mem < mean class)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_POSSIBLE, markeredgewidth=1.4, markersize=7,
               label="Mem. between classes (mean class ≤ mem < mean block)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_BIAS,     markeredgewidth=1.4, markersize=7,
               label="Mem. beyond units (mem ≥ mean units)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_UNKNOWN,  markeredgewidth=1.4, markersize=7,
               label="Unknown"),
    ]

    leg1 = ax.legend(handles=det_handles, title="Detector (mean F1)",
                     loc="upper left", bbox_to_anchor=(1.01, 1),
                     fontsize=8, title_fontsize=8,
                     framealpha=0.92, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=bias_handles, title="Memory scale (edge colour)",
              loc="lower left", bbox_to_anchor=(1.01, 0),
              fontsize=8, title_fontsize=8,
              framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    stem = f"phi_b_all_detectors_{scenario_label}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] phi_b all-detectors strip → {out_dir / stem}.{{pdf,png}}")


# ============================================================
# FIGURE 1b — phi_b strip plot (per-dataset best, 2nd, worst)
# Each point uses the detector that was best/worst FOR THAT DATASET
# (same logic as radar_base_delta.py — extreme_in_cols per dataset)
# ============================================================

def per_dataset_extreme_detector(scenario_folder: str, mode: str) -> dict:
    """
    For each dataset, finds the detector with best or worst mean F1
    in the default configuration of scenario_folder.
    Returns {dataset: detector_name}
    """
    cat = "C1" if scenario_folder == "drift" else "C3"
    def_cols = [c for c, m in col_meta.items() if m[0] == cat and not m[2]]
    result = {}
    for base in datasets:
        target_f1  = -np.inf if mode == "best" else np.inf
        target_det = None
        for col in def_cols:
            if col not in mean_df.columns:
                continue
            v = mean_df.at[base, col] if base in mean_df.index else np.nan
            if pd.notna(v):
                if (mode == "best"  and v > target_f1) or                    (mode == "worst" and v < target_f1):
                    target_f1  = v
                    target_det = col_meta[col][1]   # detector name
        result[base] = target_det
    return result


def phi_b_strip_plot_top3(dfs: dict, scenario_label: str, scenario_folder: str,
                           ranked: list, out_dir: Path):
    """
    Strip plot — per-dataset best and worst detector.
    Role is encoded by FIXED shape + colour (independent of detector):
      Best  → blue  circle    ●
      Worst → red   triangle  ▼
    Detector name is annotated beside each point.
    Edge colour encodes the 3-level bias classification.
    """
    # Fixed visual encoding per role
    ROLE_STYLE = {
        "Best" : ("#0C447C", "o", +0.15),   # blue  circle
        "Worst": ("#A8280A", "v", -0.15),   # red   triangle-down
    }

    # Per-dataset extreme detectors
    best_per_ds   = per_dataset_extreme_detector(scenario_folder, "best")
    worst_per_ds  = per_dataset_extreme_detector(scenario_folder, "worst")

    cat      = "C1" if scenario_folder == "drift" else "C3"
    def_cols = [c for c, m in col_meta.items() if m[0] == cat and not m[2]]

    all_bases = sorted(
        set(b for df in dfs.values() for b in df.index),
        key=lambda x: ABBREV.get(x, x)
    )
    labels = [ABBREV.get(b, b) for b in all_bases]
    n = len(labels)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(9.0, max(4, n * 0.60 + 2.0)))

    for i in range(n):
        ax.axhspan(i - 0.48, i + 0.48,
                   color="#F4F4F2" if i % 2 == 0 else "white", zorder=0)
    for exp in range(-7, 1):
        ax.axvline(10**exp, color="#CCCCCC", lw=0.6, ls="--", zorder=1)

    for i, b in enumerate(all_bases):
        for role, det_map in [("Best", best_per_ds),
                              ("Worst", worst_per_ds)]:
            det = det_map.get(b)
            if det is None:
                continue
            df  = dfs.get(det)
            pv  = _phi_b(df, b) if df is not None else np.nan
            ec  = bias_level(pv, DATASET_STATS.get(b))
            color, marker, off = ROLE_STYLE[role]

            ax.scatter(
                pv if pd.notna(pv) else np.nan,
                y[i] + off,
                color=color, s=70, zorder=3,
                edgecolors=ec, linewidths=1.8,
                marker=marker
            )
            if pd.notna(pv):
                # Detector name + phi_b value beside the marker
                ax.annotate(
                    f"{DET_SHORT.get(det, det)}  {pv:.1e}",
                    (pv, y[i] + off),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=6.0, color=color, va="center",
                    fontweight="bold"
                )

    ax.set_xscale("log")
    ax.set_xlim(5e-8, 5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\phi_b$ (log scale)", fontsize=10)
    ax.set_title(
        rf"$\phi_b$ per dataset — {scenario_label} (default)",
        fontsize=11, pad=10
    )
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.grid(axis="x", which="minor", color="#EEEEEE", lw=0.4, ls=":")

    # Role legend
    role_handles = [
        Line2D([0],[0], marker=ROLE_STYLE[r][1], color="w",
               markerfacecolor=ROLE_STYLE[r][0], markeredgecolor="white",
               markersize=9, label=f"{r} detector")
        for r in ["Best", "Worst"]
    ]

    # Bias legend
    bias_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_NONE,     markeredgewidth=1.8, markersize=7,
               label="Mem. within class (mem < mean class)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_POSSIBLE, markeredgewidth=1.8, markersize=7,
               label="Mem. between classes (mean class ≤ mem < mean block)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_BIAS,     markeredgewidth=1.8, markersize=7,
               label="Mem. beyond units (mem ≥ mean units)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor=EDGE_UNKNOWN,  markeredgewidth=1.8, markersize=7,
               label="Unknown"),
    ]

    leg1 = ax.legend(handles=role_handles, title="Role",
                     loc="lower right", fontsize=8, title_fontsize=8,
                     framealpha=0.95, edgecolor="#cccccc")
    ax.add_artist(leg1)
    ax.legend(handles=bias_handles, title="Memory scale (edge colour)",
              loc="upper right", fontsize=8, title_fontsize=8,
              framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()
    stem = f"phi_b_strip_{scenario_label}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] phi_b strip → {out_dir / stem}.{{pdf,png}}")

def _get_vals(df, col, all_bases):
    return [_get(df, col, b) for b in all_bases]


def rf_panel(df_best, df_worst, best_det, worst_det,
             scenario_label, out_dir):
    COLOR_BEST  = DET_STYLE[best_det][0]
    COLOR_WORST = DET_STYLE[worst_det][0]

    all_bases = sorted(
        set(df_best.index) | set(df_worst.index),
        key=lambda x: ABBREV.get(x, x)
    )
    labels = [ABBREV.get(b, b) for b in all_bases]
    n = len(labels)
    y_pos = np.arange(n)

    ncols = 2
    nrows = (len(RF_PARAMS) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.8 + 0.8))
    axes_flat = axes.flatten()

    for idx, (col, display_name, dtype, categories) in enumerate(RF_PARAMS):
        ax = axes_flat[idx]
        vals_b = _get_vals(df_best,  col, all_bases)
        vals_w = _get_vals(df_worst, col, all_bases)

        for i in range(n):
            ax.axhspan(i - 0.45, i + 0.45,
                       color="#F4F4F2" if i % 2 == 0 else "white", zorder=0)

        if dtype in ("int", "float"):
            ax.scatter(vals_b, y_pos + 0.13, color=COLOR_BEST,  s=55, zorder=3,
                       edgecolors="white", linewidths=0.5,
                       marker=DET_STYLE[best_det][1],
                       label=DET_SHORT[best_det])
            ax.scatter(vals_w, y_pos - 0.13, color=COLOR_WORST, s=55, zorder=3,
                       edgecolors="white", linewidths=0.5,
                       marker=DET_STYLE[worst_det][1],
                       label=DET_SHORT[worst_det])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(display_name, fontsize=9)
            ax.grid(axis="x", color="#DDDDDD", lw=0.5, ls="--", zorder=1)
            for i, b in enumerate(all_bases):
                vb, vw = vals_b[i], vals_w[i]
                if pd.notna(vb):
                    ax.annotate(str(int(round(vb))) if dtype == "int" else f"{vb:.2f}",
                                (vb, i + 0.13), textcoords="offset points",
                                xytext=(4, 0), fontsize=6.5,
                                color=COLOR_BEST, va="center")
                if pd.notna(vw):
                    ax.annotate(str(int(round(vw))) if dtype == "int" else f"{vw:.2f}",
                                (vw, i - 0.13), textcoords="offset points",
                                xytext=(4, -1), fontsize=6.5,
                                color=COLOR_WORST, va="center")
        else:
            str_b = [str(v) if pd.notna(v) else None for v in vals_b]
            str_w = [str(v) if pd.notna(v) else None for v in vals_w]
            if categories is None:
                categories = sorted(set(
                    [v for v in str_b if v] + [v for v in str_w if v]))
            counts_b = [sum(1 for v in str_b if v == str(c)) for c in categories]
            counts_w = [sum(1 for v in str_w if v == str(c)) for c in categories]
            x = np.arange(len(categories))
            bw = 0.35
            ax.bar(x - bw/2, counts_b, bw, color=COLOR_BEST,  alpha=0.85,
                   label=DET_SHORT[best_det])
            ax.bar(x + bw/2, counts_w, bw, color=COLOR_WORST, alpha=0.85,
                   label=DET_SHORT[worst_det])
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in categories], fontsize=9)
            ax.set_ylabel("# datasets", fontsize=9)
            ax.set_xlabel(display_name, fontsize=9)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.grid(axis="y", color="#DDDDDD", lw=0.5, ls="--", zorder=0)
            ax.set_yticks(range(0, n + 1))
            for bar_x, counts in [(x - bw/2, counts_b), (x + bw/2, counts_w)]:
                for xi, cnt in zip(bar_x, counts):
                    if cnt > 0:
                        ax.text(xi, cnt + 0.1, str(cnt), ha="center",
                                va="bottom", fontsize=8)

        ax.set_title(display_name, fontsize=10, pad=6, color="#333330")
        ax.legend(fontsize=8, framealpha=0.88, edgecolor="#cccccc",
                  loc="upper right" if dtype in ("int","float") else "best")

    for ax in axes_flat[len(RF_PARAMS):]:
        ax.set_visible(False)

    fig.suptitle(
        f"RF hyperparameters — {scenario_label} (default) | "
        f"{DET_SHORT[best_det]} (best) vs {DET_SHORT[worst_det]} (worst)",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    stem = f"rf_hyperparams_{scenario_label}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] RF panel → {out_dir / stem}.{{pdf,png}}")

# ============================================================
# CSV SUMMARY
# ============================================================

def save_summary_csv(dfs, ranked, scenario_label, out_dir):
    all_bases = sorted(
        set(b for df in dfs.values() for b in df.index),
        key=lambda x: ABBREV.get(x, x)
    )
    param_cols = ["phi_b", "scaler_detectors", "scaler_model",
                  "rf_n_estimators", "rf_max_depth", "rf_min_samples_split",
                  "rf_min_samples_leaf", "rf_max_features", "rf_bootstrap",
                  "best_macro_f1"]
    rows = []
    rank_map = {det: rank+1 for rank, (det, _) in enumerate(ranked)}
    for b in all_bases:
        stats = DATASET_STATS.get(b)
        for det in ALL_DETECTORS:
            df = dfs.get(det)
            row = {
                "dataset"    : ABBREV.get(b, b),
                "detector"   : DET_SHORT[det],
                "rank"       : rank_map.get(det, np.nan),
                "min_seg"    : stats["min_seg"]    if stats else np.nan,
                "median_seg" : stats["median_seg"] if stats else np.nan,
                "max_seg"    : stats["max_seg"]    if stats else np.nan,
                "block_size" : stats["block_size"] if stats else np.nan,
            }
            for c in param_cols:
                row[c] = _get(df, c, b) if df is not None else np.nan
            phi = row.get("phi_b")
            if phi and not (isinstance(phi, float) and np.isnan(phi)) and stats:
                mem = memory_instances(phi)
                row["memory_instances"] = mem
                row["bias_level"] = {
                    EDGE_NONE    : "within_class",
                    EDGE_POSSIBLE: "between_classes",
                    EDGE_BIAS    : "possible_bias",
                    EDGE_UNKNOWN : "unknown",
                }.get(bias_level(phi, stats), "unknown")
            else:
                row["memory_instances"] = np.nan
                row["bias_level"]       = "unknown"
            rows.append(row)
    path = out_dir / f"hyperparams_summary_{scenario_label}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[OK] Summary CSV → {path}")

# ============================================================
# MAIN LOOP
# ============================================================

for clabel, folder in SCENARIOS.items():
    print(f"\n{'='*70}")
    print(f"SCENARIO {clabel} — folder: {folder}")
    print('='*70)

    ranked = rank_detectors(clabel)
    best_det,  best_mean  = ranked[0]
    worst_det, worst_mean = ranked[-1]

    second_det, second_mean = ranked[1]

    print(f"Detector ranking (mean {clabel} default F1):")
    for det, v in ranked:
        tag = " ← BEST"   if det == best_det   else \
              " ← 2ND"    if det == second_det  else \
              " ← WORST"  if det == worst_det   else ""
        print(f"  {DET_SHORT[det]:8s}  {v:.4f}{tag}")

    # Load all detectors
    dfs = {}
    for det in ALL_DETECTORS:
        print(f"  Loading {det}...")
        dfs[det] = load_params(datasets, folder, det)

    phi_b_strip_plot_all(dfs, clabel, ranked, OUTPUT_DIR)
    phi_b_strip_plot_top3(dfs, clabel, folder, ranked, OUTPUT_DIR)
    rf_panel(dfs[best_det], dfs[worst_det], best_det, worst_det,
             clabel, OUTPUT_DIR)
    save_summary_csv(dfs, ranked, clabel, OUTPUT_DIR)

print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")
# ============================================================
# ANÁLISE DE RIGIDEZ DO PROTOCOLO
# Hipótese: se o ganho de C3 vs C2 viesse de memorizar a sequência
# do protocolo, datasets com protocolo mais rígido (baixo CV de
# duração de classe entre blocos) deveriam mostrar ganhos maiores.
# ============================================================

def protocol_rigidity(dataset_key: str) -> dict | None:
    """
    Computes protocol rigidity metrics from block files.

    For each class, computes the duration (sample count) across all blocks
    that contain it, then calculates CV (std/mean). The mean CV across all
    classes is the rigidity index: low = rigid, high = variable.

    Also computes the order consistency: whether the sequence of class
    transitions is identical across blocks (1.0 = perfectly rigid,
    0.0 = completely variable).

    Returns dict with:
        mean_cv          : mean CV of class durations across blocks (↓ = rigid)
        order_consistency: fraction of block pairs with identical transition seq
        n_blocks         : number of blocks analysed
        class_durations  : {class_label: [duration_per_block]}
    """
    if dataset_key not in DATASET_LAYOUTS:
        return None
    subfolder, label_col, nome_base, end_base, folds = DATASET_LAYOUTS[dataset_key]
    data_dir = DATASETS_ROOT / subfolder
    all_indices = sorted({idx for fold in folds for idx in fold})

    # Per-block: class durations and transition sequence
    block_class_dur  = {}   # {idx: {class: total_samples}}
    block_transitions = {}  # {idx: tuple of (class_a → class_b)}
    found_any = False

    for idx in all_indices:
        fpath = data_dir / f"{nome_base}{idx}{end_base}"
        if not fpath.exists():
            continue
        found_any = True
        try:
            df = pd.read_csv(fpath, usecols=[label_col])
        except Exception:
            continue
        labels = df[label_col].values
        if len(labels) == 0:
            continue

        # Class durations
        dur = {}
        transitions = []
        run_len, run_cls = 1, labels[0]
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                run_len += 1
            else:
                dur[run_cls] = dur.get(run_cls, 0) + run_len
                transitions.append(run_cls)
                run_cls  = labels[i]
                run_len  = 1
        dur[run_cls] = dur.get(run_cls, 0) + run_len
        transitions.append(run_cls)

        block_class_dur[idx]   = dur
        block_transitions[idx] = tuple(transitions)

    if not found_any or len(block_class_dur) < 2:
        return None

    # CV per class across blocks
    all_classes = set(c for dur in block_class_dur.values() for c in dur)
    cvs = []
    class_durations = {}
    for cls in all_classes:
        durs = [block_class_dur[idx].get(cls, 0)
                for idx in block_class_dur]
        class_durations[str(cls)] = durs
        mean_d = np.mean(durs)
        if mean_d > 0:
            cvs.append(np.std(durs) / mean_d)

    mean_cv = float(np.mean(cvs)) if cvs else np.nan

    # Order consistency: fraction of block pairs with identical transition seq
    seqs = list(block_transitions.values())
    n_pairs = len(seqs) * (len(seqs) - 1) / 2
    identical = sum(
        1 for i in range(len(seqs))
        for j in range(i + 1, len(seqs))
        if seqs[i] == seqs[j]
    )
    order_consistency = identical / n_pairs if n_pairs > 0 else np.nan

    return {
        "mean_cv"          : mean_cv,
        "order_consistency": order_consistency,
        "n_blocks"         : len(block_class_dur),
        "class_durations"  : class_durations,
    }


def plot_rigidity_vs_gain(rigidity_stats: dict, mean_df: pd.DataFrame,
                           col_meta: dict, out_dir: Path):
    """
    Scatter plot: protocol rigidity (mean CV, lower = more rigid) vs
    F1 gain (C3_best_default − C2) per dataset.

    Two panels: left = mean_cv vs gain, right = order_consistency vs gain.
    Includes Pearson r and linear regression line.
    """
    from scipy import stats as sp_stats

    # Identify C2 column and C3 default columns
    c2_col = next((c for c, m in col_meta.items() if m[0] == "C2"), None)
    c3_cols = {m[1]: c for c, m in col_meta.items()
               if m[0] == "C3" and not m[2] and m[1] in ALL_DETECTORS}

    if c2_col is None or not c3_cols:
        print("[WARN] Could not find C2 or C3 columns for rigidity analysis")
        return

    rows = []
    for ds, rig in rigidity_stats.items():
        if rig is None:
            continue
        if ds not in mean_df.index:
            continue

        f1_c2 = mean_df.at[ds, c2_col] if c2_col in mean_df.columns else np.nan
        if pd.isna(f1_c2):
            continue

        # Best C3 default for this dataset
        best_c3 = max(
            (mean_df.at[ds, col] for col in c3_cols.values()
             if col in mean_df.columns and pd.notna(mean_df.at[ds, col])),
            default=np.nan
        )
        if pd.isna(best_c3):
            continue

        gain = best_c3 - f1_c2
        rows.append({
            "dataset"          : ABBREV.get(ds, ds),
            "mean_cv"          : rig["mean_cv"],
            "order_consistency": rig["order_consistency"],
            "gain_c3_vs_c2"    : gain,
            "f1_c2"            : f1_c2,
            "f1_c3_best"       : best_c3,
        })

    if len(rows) < 3:
        print("[WARN] Too few datasets for rigidity analysis")
        return

    df_rig = pd.DataFrame(rows)
    df_rig.to_csv(out_dir / "rigidity_vs_gain.csv", index=False)
    print(f"[OK] Rigidity CSV → {out_dir / 'rigidity_vs_gain.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, x_col, x_label, x_inv in [
        (axes[0], "mean_cv",
         "Protocol rigidity\n(mean CV of class duration, ↓ = more rigid)",
         False),
        (axes[1], "order_consistency",
         "Order consistency\n(fraction of blocks with identical sequence, ↑ = more rigid)",
         False),
    ]:
        x = df_rig[x_col].values
        y = df_rig["gain_c3_vs_c2"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_v, y_v = x[mask], y[mask]

        ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)

        # Scatter
        ax.scatter(x_v, y_v, color="#0C447C", s=70, zorder=3,
                   edgecolors="white", linewidths=0.5)

        # Dataset labels
        for i, row in df_rig[mask].iterrows():
            ax.annotate(row["dataset"], (row[x_col], row["gain_c3_vs_c2"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="#333333")

        # Regression + Pearson r
        if len(x_v) >= 3:
            slope, intercept, r, p, _ = sp_stats.linregress(x_v, y_v)
            x_line = np.linspace(x_v.min(), x_v.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#A8280A", lw=1.5, ls="-",
                    label=f"r = {r:.2f}  (p = {p:.3f})")
            ax.legend(fontsize=9, framealpha=0.9)

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("F1 gain: C3 best default − C2", fontsize=9)
        ax.grid(color="#EEEEEE", lw=0.5)

    fig.suptitle(
        "Protocol rigidity vs F1 gain (C3 − C2)\n"
        "Hypothesis: rigid protocols → larger gains if bias drives C3 improvement",
        fontsize=11
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"rigidity_vs_gain.{ext}",
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Rigidity plot → {out_dir / 'rigidity_vs_gain'}.{{pdf,png}}")


# ============================================================
# COMPUTE RIGIDITY AND PLOT
# ============================================================
print("\n[INFO] Computing protocol rigidity per dataset...")
RIGIDITY = {}
for ds in ABBREV:
    rig = protocol_rigidity(ds)
    RIGIDITY[ds] = rig
    if rig:
        print(f"  {ABBREV[ds]:12s}  mean_cv={rig['mean_cv']:.3f}  "
              f"order_consistency={rig['order_consistency']:.2f}  "
              f"n_blocks={rig['n_blocks']}")
    else:
        print(f"  {ABBREV[ds]:12s}  N/A")

plot_rigidity_vs_gain(RIGIDITY, mean_df, col_meta, OUTPUT_DIR)

# ============================================================
# ANÁLISE: ESCALA DE MEMÓRIA DO MELHOR DETECTOR vs GANHO F1
# Pergunta: datasets com memória dentro da classe (cinza) vs
# entre classes (laranja) mostram ganhos F1 diferentes?
# ============================================================

def _build_gain_df(clabel: str, folder: str, ref_col: str,
                   dfs: dict, mean_df: pd.DataFrame,
                   col_meta: dict, dataset_stats: dict) -> pd.DataFrame:
    """Builds the gain + memory-scale DataFrame for one scenario."""
    cat = clabel
    def_cols = {m[1]: c for c, m in col_meta.items()
                if m[0] == cat and not m[2] and m[1] in ALL_DETECTORS}
    level_map = {
        EDGE_NONE    : "within_class",
        EDGE_POSSIBLE: "between_classes",
        EDGE_BIAS    : "possible_bias",
        EDGE_UNKNOWN : "unknown",
    }
    rows = []
    for ds in mean_df.index:
        f1_ref = mean_df.at[ds, ref_col] if ref_col in mean_df.columns else np.nan
        if pd.isna(f1_ref):
            continue
        best_f1, best_det_ds = -np.inf, None
        for det, col in def_cols.items():
            if col not in mean_df.columns:
                continue
            v = mean_df.at[ds, col] if ds in mean_df.index else np.nan
            if pd.notna(v) and v > best_f1:
                best_f1, best_det_ds = v, det
        if best_det_ds is None or pd.isna(best_f1):
            continue
        gain    = best_f1 - f1_ref
        df_det  = dfs.get(best_det_ds)
        phi     = _phi_b(df_det, ds) if df_det is not None else np.nan
        stats   = dataset_stats.get(ds)
        level   = level_map.get(bias_level(phi, stats), "unknown")
        rows.append({
            "dataset": ABBREV.get(ds, ds),
            "gain"   : gain,
            "level"  : level,
            "phi_b"  : phi,
            "det"    : DET_SHORT.get(best_det_ds, best_det_ds),
        })
    return pd.DataFrame(rows)


def _adjust_labels(ax, points: list[tuple], color: str,
                   x_offset: float = 0.08, fontsize: float = 7.0):
    """
    Places labels at the same y as each point.
    All points are at x = xi (no jitter). Labels alternate
    right/left when consecutive points are too close in y.
    """
    if not points:
        return

    # Estimate minimum y gap for overlap (in data units)
    y_lo, y_hi = ax.get_ylim()
    fig_h = ax.get_figure().get_size_inches()[1] * ax.get_position().height
    min_y_gap = (y_hi - y_lo) / fig_h * (fontsize / 72.0) * 1.4

    # Sort by y ascending
    pts = sorted(points, key=lambda p: p[1])

    # Determine side (right=+1, left=-1) alternating when overlap detected
    sides = [1]  # first label always to the right
    for i in range(1, len(pts)):
        if abs(pts[i][1] - pts[i-1][1]) < min_y_gap:
            sides.append(-sides[i-1])  # alternate side
        else:
            sides.append(1)  # reset to right

    for (ds_name, x_data, y_data), side in zip(pts, sides):
        ha = "left" if side > 0 else "right"
        ax.text(
            x_data + side * x_offset, y_data,
            ds_name,
            fontsize=fontsize, color=color,
            va="center", ha=ha,
            clip_on=False,
        )

def sci_tex(v):
    if np.isnan(v):
        return "--"

    superscript = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

    exp = int(np.floor(np.log10(abs(v))))
    mant = v / (10 ** exp)

    return f"{mant:.1f} × 10{str(exp).translate(superscript)}"

def _plot_scenario(ax, df_plot: pd.DataFrame, clabel: str, gain_label: str,
                   GROUP_ORDER, GROUP_COLOR, GROUP_LABEL):
    """
    Strip plot com rótulos ajustados para evitar sobreposição.

    Correções nesta versão:
      1. A barra da mediana fica atrás dos pontos e dos rótulos.
      2. Cada rótulo recebe uma pequena caixa branca, evitando que a barra
         da mediana ou as linhas de ligação escondam o texto.
      3. As linhas de ligação ficam com zorder baixo e não cobrem os nomes.
      4. Quando o ponto está muito próximo da mediana, o rótulo já nasce
         levemente deslocado em y, reduzindo a chance de ficar sobre a barra.
    """
    from adjustText import adjust_text
    print("------------", clabel)
    x_pos = {g: i for i, g in enumerate(GROUP_ORDER)}
    BAR_W = 0.44
    FS = 12.0

    all_texts = []

    # Distância vertical mínima para afastar rótulos que caem exatamente
    # sobre a barra da mediana. É calculada em unidades do eixo y.
    y_lo, y_hi = ax.get_ylim()
    y_span = max(y_hi - y_lo, 1e-9)
    median_label_gap = 0.018 * y_span

    for group in GROUP_ORDER:
        sub = df_plot[df_plot["level"] == group].copy()
        if sub.empty:
            continue

        xi = x_pos[group]
        sub = sub.sort_values("gain").reset_index(drop=True)

        # Pequeno espalhamento horizontal determinístico para reduzir
        # sobreposição inicial entre pontos e rótulos.
        n_sub = len(sub)

        if n_sub == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.10, 0.10, n_sub)

        xs = np.full(n_sub, xi, dtype=float) + offsets

        med = sub["gain"].median()

        # A mediana deve ficar atrás dos pontos e, principalmente, atrás dos textos.
        ax.plot([xi - BAR_W, xi + BAR_W], [med, med],
                color=GROUP_COLOR[group], lw=2.5, zorder=1,
                solid_capstyle="butt")

        ax.scatter(
            xs, sub["gain"],
            color=GROUP_COLOR[group], s=70, zorder=4,
            edgecolors="white", linewidths=0.5, alpha=0.9
        )

        for k, (x_pt, (_, row)) in enumerate(zip(xs, sub.iterrows())):
            y_pt = float(row["gain"])

            # Se o rótulo cair muito perto da mediana, desloca o texto
            # para cima/baixo alternadamente. Isso evita que nomes como
            # SmPh, P2-Chest, RO etc. fiquem escondidos na barra.
            dy = 0.0
            if abs(y_pt - med) < median_label_gap:
                dy = median_label_gap * (1 if k % 2 == 0 else -1)

            t = ax.text(
                x_pt + 0.055, y_pt + dy,
                row["dataset"],
                fontsize=FS,
                color=GROUP_COLOR[group],
                va="center",
                ha="left",
                clip_on=False,
                zorder=10,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                    boxstyle="round,pad=0.12"
                )
            )
            all_texts.append(t)

    ax.set_title(f"{clabel}: {gain_label}", fontsize=14)
    def _fmt_phi_stats(group: str) -> str:
        vals = pd.to_numeric(
            df_plot.loc[df_plot["level"] == group, "phi_b"],
            errors="coerce"
        ).dropna()

        if len(vals) == 0:
            return r"$\phi_b$: --"

        min_phi = vals.min()
        mean_phi = vals.mean()
        max_phi = vals.max()

        return (
            #r"$\phi_b$: "+ 
            f"[{sci_tex(min_phi)}, {sci_tex(max_phi)}]" #< {sci_tex(mean_phi)} 
        )

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [f"{GROUP_LABEL[g]}\n{_fmt_phi_stats(g)}"
         for g in GROUP_ORDER],
        fontsize=13
    )
    ax.set_ylabel("F1 gain (percentage points)", fontsize=13)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=0)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5, zorder=0)

    # Ajusta textos após todos os pontos serem desenhados.
    # As linhas são finas, claras e ficam atrás das caixas dos textos.
    adjust_text(
        all_texts,
        ax=ax,
        only_move={"points": "y", "text": "xy", "objects": "y"},
        expand_points=(1.5, 1.8),
        expand_text=(1.4, 1.8),
        force_points=(0.25, 0.45),
        force_text=(0.45, 0.90),
        lim=300,
        arrowprops=dict(
            arrowstyle="-",
            color="#9A9A9A",
            lw=0.35,
            alpha=0.75,
            shrinkA=6,
            shrinkB=4,
            zorder=2,
            connectionstyle="arc3,rad=0.0"
        )
    )

    # Garante que, mesmo após o adjustText, os textos fiquem acima de tudo.
    for t in all_texts:
        t.set_zorder(10)

def plot_memory_scale_vs_gain(dfs_by_scenario: dict, mean_df: pd.DataFrame,
                               col_meta: dict, dataset_stats: dict,
                               out_dir: Path):
    """
    Produces two separate figures (C1 and C3), one per scenario, and a
    third combined figure with C1 and C3 stacked vertically.

    The individual C1 and C3 panels are generated by the same
    _plot_scenario() function used in the separate figures, preserving
    fonts, labels, markers, colours, medians and annotation behaviour.
    """
    c0_col = next((c for c, m in col_meta.items() if m[0] == "C0"), None)
    c2_col = next((c for c, m in col_meta.items() if m[0] == "C2"), None)

    # Paleta LOCAL desta figura (distinta das constantes globais EDGE_NONE/
    # EDGE_POSSIBLE/EDGE_BIAS, que permanecem inalteradas para as demais
    # figuras do script). Tons da família de azuis do artigo (ancorada em
    # #355C7D), do mais escuro ao mais claro, em vez de cinza/laranja/
    # vermelho — evita confusão com a paleta de magnitude do Cliff's delta,
    # que usa essas mesmas cores com outro significado em outras figuras.
    GROUP_COLOR = {
        "within_class"   : "#16263A",  # mais escuro: memoria curta
        "between_classes": "#3E6E94",  # tom medio
        "possible_bias"  : "#4A7A99",  # memoria longa (beyond units) — contraste
                                        # verificado (~4.6:1 contra branco, acima
                                        # do minimo WCAG de 4.5:1 para texto normal)
        "unknown"        : "#AAAAAA",
    }
    GROUP_LABEL = {
        "within_class"   : "Mem. within class",
        "between_classes": "Mem. between classes",
        "possible_bias"  : "Mem. beyond units",
    }
    GROUP_ORDER = ["within_class", "between_classes", "possible_bias"]

    scenario_configs = [
        ("C1", "drift",         c0_col, "C1 best − C0 (baseline)"),
        ("C3", "catch24_drift", c2_col, "C3 best − C2 (catch22)"),
    ]

    # Cache the exact data and y-limits used in the individual figures.
    # The combined figure below reuses these objects so that panels (a)
    # and (b) match the separately saved C1 and C3 figures.
    panel_cache = []
    aux = ['(a) ', '(b) ']
    indice = 0
    for clabel, folder, ref_col, gain_label in scenario_configs:
        if ref_col is None:
            continue
        dfs     = dfs_by_scenario.get(folder, {})
        df_plot = _build_gain_df(clabel, folder, ref_col, dfs,
                                  mean_df, col_meta, dataset_stats)
        if df_plot.empty:
            continue

        # Save CSV
        df_plot.to_csv(out_dir / f"memory_scale_vs_gain_{clabel}.csv", index=False)
        print(f"\n[CONFERIR com a Tabela S-15] {clabel}: ganho do melhor detector (p.p.)")
        for _, r in df_plot.sort_values("level").iterrows():
            print(f"  {r['dataset']:10s} {r['level']:16s} {r['det']:6s} "
                  f"gain={r['gain']:6.2f}  phi_b={r['phi_b']:.2e}")
        for g, sub in df_plot.groupby("level"):
            print(f"  grupo {g:16s} n={len(sub):2d}  mediana={sub['gain'].median():.2f}")

        # Determine y-axis range with margin for labels
        y_max = df_plot["gain"].max() * 1.15
        y_min = min(df_plot["gain"].min() - 1, -1)

        panel_cache.append({
            "clabel": clabel,
            "gain_label": gain_label,
            "df_plot": df_plot,
            "y_min": y_min,
            "y_max": y_max,
        })

        fig, ax = plt.subplots(figsize=(8.2, 6.8))

        # Os limites precisam ser definidos antes de chamar _plot_scenario(),
        # pois o adjustText usa os limites atuais para reposicionar os rótulos.
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.65, len(GROUP_ORDER) - 0.35)

        _plot_scenario(ax, df_plot,  clabel, gain_label,
                       GROUP_ORDER, GROUP_COLOR, GROUP_LABEL)
        
        fig.suptitle(
            "Memory scale of best detector vs F1 gain\n"
            "Horizontal line = group median",
            fontsize=14
        )
        fig.tight_layout()
        stem = f"memory_scale_vs_gain_{clabel}"
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}.{ext}",
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[OK] Memory scale vs gain {clabel} → {out_dir / stem}.{{pdf,png}}")

    # ------------------------------------------------------------
    # Combined figure: C1 panel (a) over C3 panel (b)
    # ------------------------------------------------------------
    if len(panel_cache) >= 2:
        fig, axes = plt.subplots(
            nrows=2, ncols=1,
            figsize=(8.2, 11.5),
            sharex=False
        )

        for ax, panel, panel_letter in zip(axes, panel_cache[:2], ["", ""]):
            ax.set_ylim(panel["y_min"], panel["y_max"])
            ax.set_xlim(-0.65, len(GROUP_ORDER) - 0.35)

            _plot_scenario(
                ax, panel["df_plot"], aux[indice] + panel["clabel"], panel["gain_label"],
                GROUP_ORDER, GROUP_COLOR, GROUP_LABEL
            )
            indice += 1
            
            # Panel identifier. Kept outside the plotting area to avoid
            # changing the content of the original C1/C3 panels.
            ax.text(
                -0.08, 1.05, panel_letter,
                transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                ha="left", va="bottom",
                clip_on=False
            )

        fig.tight_layout()

        stem = "memory_scale_vs_gain_C1_C3_vertical"
        for ext in ("pdf", "png"):

            fig.savefig(out_dir / f"{stem}.{ext}",
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"[OK] Combined C1/C3 vertical figure → {out_dir / stem}.{{pdf,png}}")
    else:
        print("[WARN] Combined C1/C3 figure not generated: fewer than two panels available.")


# Collect dfs per scenario folder from main loop results
# Re-load if needed (dfs is only available inside the loop above)
# Build a cache at the bottom of the main loop by re-running load_params
print("\n[INFO] Loading params for memory-scale vs gain analysis...")
dfs_by_scenario = {}
for clabel, folder in SCENARIOS.items():
    dfs_by_scenario[folder] = {
        det: load_params(datasets, folder, det)
        for det in ALL_DETECTORS
    }

plot_memory_scale_vs_gain(
    dfs_by_scenario, mean_df, col_meta, DATASET_STATS, OUTPUT_DIR
)

# ============================================================
# GANHO vs HORIZONTE DE MEMÓRIA / TAMANHO DA UNIDADE
# h = horizonte da eq:horizonte (S_j cai a e^{-1}), o mesmo da Seção 4.
# Eixo x: h / tamanho médio da unidade experimental (log), linha em 1.
# Eixo y: ganho do melhor detector. (a) C1 - C0, (b) C3 - C2.
# Descritivo: 16 pontos, melhor detector escolhido a posteriori, sem teste.
# ============================================================

def horizon_instances(phi_b: float, k_max: int = 20_000_000, chunk: int = 1_000_000) -> int:
    """Menor h com sum_{k=1}^{h} -ln(1 - min(1, phi_b ln(1+k))) >= 1 (eq:horizonte)."""
    if phi_b is None or not np.isfinite(phi_b) or phi_b <= 0:
        return np.nan
    acc, k0 = 0.0, 1
    while k0 <= k_max:
        k = np.arange(k0, min(k0 + chunk, k_max + 1), dtype=float)
        phi = np.minimum(1.0, phi_b * np.log1p(k))
        if np.any(phi >= 1.0):                      # saturação: S_j zera
            idx = int(np.argmax(phi >= 1.0))
            soma = acc + np.cumsum(-np.log1p(-phi[:idx]))
            hit = np.flatnonzero(soma >= 1.0)
            return int(k[hit[0]]) if hit.size else int(k[idx])
        soma = acc + np.cumsum(-np.log1p(-phi))
        hit = np.flatnonzero(soma >= 1.0)
        if hit.size:
            return int(k[hit[0]])
        acc = float(soma[-1]); k0 += chunk
    return k_max


COR_PONTO, COR_TEXTO, COR_LINHA = "#355C7D", "#16263A", "#7A7A7A"
# rotulos colocados a esquerda do ponto para nao encostar em pontos vizinhos
ESQUERDA = {"C1": {"W-LAnkle", "W-LWrist", "P2-Chest", "W-RWrist"}, "C3": {"P2-Chest"}}


def _rotulos(ax, d, gap, esquerda=()):
    """Rotulo a direita do ponto; sobreposicoes resolvidas so na vertical,
    entre rotulos cujos pontos estao proximos no eixo x (log)."""
    pts = d[["h_over_unit", "gain", "dataset"]].copy()
    pts["lx"] = np.log10(pts["h_over_unit"])
    pts["ty"] = pts["gain"].astype(float)
    pts["esq"] = pts["dataset"].isin(esquerda)
    pts = pts.sort_values("ty").reset_index(drop=True)
    for _ in range(200):
        mudou = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if (pts.esq[i] == pts.esq[j] and abs(pts.lx[i] - pts.lx[j]) < 0.55
                        and pts.ty[j] - pts.ty[i] < gap):
                    meio = (pts.ty[i] + pts.ty[j]) / 2
                    pts.loc[i, "ty"] = meio - gap / 2 - 1e-6
                    pts.loc[j, "ty"] = meio + gap / 2 + 1e-6
                    mudou = True
        if not mudou:
            break
    for _, r in pts.iterrows():
        x, y, ty = r.h_over_unit, r.gain, r.ty
        esq = r.dataset in esquerda
        tx = x / 1.35 if esq else x * 1.35
        ax.annotate(r.dataset, xy=(x, y), xytext=(tx, ty), fontsize=12, color=COR_TEXTO,
                    va="center", ha="right" if esq else "left", zorder=4,
                    arrowprops=None if abs(ty - y) < gap * 0.35 else
                    dict(arrowstyle="-", color="#9A9A9A", lw=0.5, shrinkA=0, shrinkB=2.5))


def _plot_horizonte(t, saida):
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 9.2), sharex=True)
    titulos = {"C1": "(a) C1 best $-$ C0", "C3": "(b) C3 best $-$ C2 (catch22)"}
    for ax, cen in zip(axs, ["C1", "C3"]):
        d = t[t.scenario == cen]
        ymax = d.gain.max()
        gap = 0.062 * (ymax * 1.12 + 1.5)
        ax.axvspan(1.0, 100, color="#F2F2F2", zorder=0, lw=0)
        ax.axvline(1.0, color=COR_LINHA, lw=1.0, ls="--", zorder=1)
        ax.axhline(0.0, color="#C8C8C8", lw=0.8, zorder=1)
        dentro = d.h < d.class_seg
        ax.scatter(d.h_over_unit[dentro], d.gain[dentro], s=54, marker="o", c=COR_PONTO,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="$h$ shorter than the mean class segment")
        ax.scatter(d.h_over_unit[~dentro], d.gain[~dentro], s=64, marker="^", c="#C0504D",
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="$h$ longer than the mean class segment")
        _rotulos(ax, d, gap, ESQUERDA.get(cen, ()))
        ax.set_xscale("log")
        ax.set_xlim(3e-4, 100)
        ax.set_ylim(-1.5, ymax * 1.12)
        ax.set_title(titulos[cen], loc="left", fontsize=14)
        ax.set_ylabel("F1 gain (percentage points)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(axis="both", color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(1.15, -0.8, "$h$ beyond the unit", fontsize=11, color=COR_LINHA,
                ha="left", va="center")
    h_, l_ = axs[1].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               fontsize=12, handletextpad=0.3, columnspacing=1.2)
    axs[-1].set_xlabel("Memory horizon $h$ / mean size of the experimental unit", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{saida}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_horizon_vs_gain(dfs_by_scenario: dict, mean_df: pd.DataFrame,
                         col_meta: dict, dataset_stats: dict, out_dir: Path):
    c0_col = next((c for c, m in col_meta.items() if m[0] == "C0"), None)
    c2_col = next((c for c, m in col_meta.items() if m[0] == "C2"), None)
    configs = [("C1", "drift", c0_col, "(a) C1 best $-$ C0"),
               ("C3", "catch24_drift", c2_col, "(b) C3 best $-$ C2 (catch22)")]
    inv = {v: k for k, v in ABBREV.items()}
    tabelas = []
    for clabel, folder, ref_col, titulo in configs:
        if ref_col is None:
            continue
        d = _build_gain_df(clabel, folder, ref_col, dfs_by_scenario.get(folder, {}),
                           mean_df, col_meta, dataset_stats)
        if d.empty:
            continue
        d["h"] = [horizon_instances(p) for p in d["phi_b"]]
        d["unit"] = [(dataset_stats.get(inv.get(n, n)) or {}).get("block_size", np.nan)
                     for n in d["dataset"]]
        d["class_seg"] = [(dataset_stats.get(inv.get(n, n)) or {}).get("mean_seg", np.nan)
                          for n in d["dataset"]]
        d["h_over_unit"] = d["h"] / d["unit"]
        d["scenario"], d["titulo"] = clabel, titulo
        tabelas.append(d)
    if not tabelas:
        print("[WARN] horizon vs gain: sem dados")
        return
    t = pd.concat(tabelas, ignore_index=True)
    t.to_csv(out_dir / "horizon_vs_gain_C1_C3.csv", index=False)
    print("\n[CONFERIR] ganho vs h/unidade")
    for _, r in t.iterrows():
        print(f"  {r['scenario']} {r['dataset']:10s} {r['det']:6s} phi_b={r['phi_b']:.2e} "
              f"h={r['h']:>9,.0f} unit={r['unit']:>9,.0f} h/unit={r['h_over_unit']:.3g} "
              f"gain={r['gain']:6.2f}")

    _plot_horizonte(t, out_dir / "horizon_vs_gain_C1_C3")
    return t
    print(f"[OK] Horizon vs gain → {out_dir / 'horizon_vs_gain_C1_C3'}.{{pdf,png}}")


t_horizonte = plot_horizon_vs_gain(dfs_by_scenario, mean_df, col_meta, DATASET_STATS, OUTPUT_DIR)


# ============================================================
# SmPh: DURAÇÃO DAS CLASSES DE TRANSIÇÃO vs h
# As 6 classes de transição postural são as 6 de maior rótulo (7 a 12 no
# HAPT original). O script imprime a duração de todas as classes para conferir.
# ============================================================

def smph_transition_stats(t: pd.DataFrame | None, out_dir: Path, n_transicoes: int = 6):
    subfolder, label_col, nome_base, end_base, folds = DATASET_LAYOUTS["smartphone"]
    data_dir = DATASETS_ROOT / subfolder
    runs = {}                                # rótulo -> lista de durações dos segmentos
    for idx in sorted({i for f in folds for i in f}):
        fpath = data_dir / f"{nome_base}{idx}{end_base}"
        if not fpath.exists():
            continue
        y = pd.read_csv(fpath, usecols=[label_col])[label_col].to_numpy()
        if len(y) == 0:
            continue
        ini = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1]
        fim = np.r_[ini[1:], len(y)]
        for a, b in zip(ini, fim):
            runs.setdefault(y[a], []).append(b - a)
    if not runs:
        print("[WARN] SmPh: arquivos não encontrados")
        return
    linhas = [{"classe": c, "segmentos": len(v), "media": float(np.mean(v)),
               "mediana": float(np.median(v)), "min": int(np.min(v)), "max": int(np.max(v))}
              for c, v in sorted(runs.items())]
    tab = pd.DataFrame(linhas)
    rot_tr = sorted(runs)[-n_transicoes:]
    tab["transicao"] = tab["classe"].isin(rot_tr)
    tab.to_csv(out_dir / "smph_duracao_classes.csv", index=False)
    print("\n[SmPh] duração dos segmentos por classe (amostras)")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    tr = np.concatenate([runs[c] for c in rot_tr])
    at = np.concatenate([runs[c] for c in runs if c not in rot_tr])
    print(f"  transições (rótulos {rot_tr}): {len(tr)} segmentos, "
          f"média {tr.mean():,.1f}, mediana {np.median(tr):,.1f}")
    print(f"  atividades: {len(at)} segmentos, média {at.mean():,.1f}, mediana {np.median(at):,.1f}")
    if t is not None:
        for _, r in t[t["dataset"] == "SmPh"].iterrows():
            print(f"  {r['scenario']}: h = {r['h']:,.0f}  ->  h / transição média = "
                  f"{r['h'] / tr.mean():.2f},  h / transição mediana = {r['h'] / np.median(tr):.2f}")


smph_transition_stats(t_horizonte, OUTPUT_DIR)