from pathlib import Path
from functools import lru_cache
from copy import deepcopy

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from tabs.decomp_variabilidad import (
    decomposition_card,
    register_decomposition_callbacks,
)

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

METRIC_OPTIONS = [
    {"label": "SD", "value": "SD"},
    {"label": "RMSE", "value": "RMSE"},
    {"label": "MAE", "value": "MAE"},
    {"label": "MSE", "value": "MSE"},
]

COMMON_GATE_COLS = ["Rrup_km", "Magnitude", "Hypocenter Depth (km)"]

GATE_GRID_N = 60
GATE_MAX_POINTS = 1200
DEFAULT_ALPHA = 4.0
DEFAULT_BETA = 3.7184
DEFAULT_SOIL_CONF = 1.0
SOIL_COL = "Soil_Class"


# =========================================================
# ESTILO VISUAL
# =========================================================

COLORS = {
    "base": "#1F2937",
    "oof": "#2563EB",
    "filtered": "#0F766E",
    "full": "#7C3AED",
    "full_gated": "#B45309",
    "reference": "#6B7280",
    "zero": "#111827",
    "hist": "#334155",
    "hist_line": "#0F172A",
    "scatter": "#1E3A8A",
    "ols": "#B45309",
    "points": "#F59E0B",
    "points_border": "#111827",
}

SERIOUS_COLORWAY = [
    COLORS["base"],
    COLORS["oof"],
    COLORS["filtered"],
    COLORS["full"],
    COLORS["full_gated"],
    "#475569",
    "#64748B",
]

GATE_COLORSCALE = [
    [0.00, "#0F172A"],
    [0.20, "#334155"],
    [0.40, "#64748B"],
    [0.60, "#94A3B8"],
    [0.80, "#CBD5E1"],
    [1.00, "#F8FAFC"],
]


def apply_common_layout(fig, title=None, height=460):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=30, r=25, t=65, b=75),
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            size=13,
            color="#1F2937",
        ),
        title_font=dict(size=18, color="#111827"),
        colorway=SERIOUS_COLORWAY,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0,
            font=dict(size=12),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#374151"),
        title_font=dict(color="#111827"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#374151"),
        title_font=dict(color="#111827"),
    )

    return fig


def empty_figure(message, height=460):
    fig = go.Figure()
    fig.add_annotation(
        text=str(message),
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color="#334155"),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_visible=False,
        yaxis_visible=False,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# =========================================================
# CARGA DEL PAQUETE
# =========================================================

@lru_cache(maxsize=1)
def get_package():
    try:
        from model.train_model import load_model_package
        return load_model_package("model/model.pkl")
    except Exception as exc:
        return exc


def model_options(package):
    if isinstance(package, Exception):
        return []

    return [
        {"label": artifact.get("label", key), "value": key}
        for key, artifact in package.get("models", {}).items()
    ]


def period_options(artifact):
    periods = np.asarray(artifact.get("periods", []), dtype=float)
    return [{"label": f"T = {p:g} s", "value": float(p)} for p in periods]


def selected_period_index(artifact, period):
    periods = np.asarray(artifact["periods"], dtype=float)

    if period is None:
        return 0

    return int(np.argmin(np.abs(periods - float(period))))


# =========================================================
# GATE COMÚN
# =========================================================

def _subset_standard_scaler(scaler, old_cols, new_cols):
    new_scaler = deepcopy(scaler)

    old_cols = list(old_cols)
    new_cols = list(new_cols)

    idx = [old_cols.index(c) for c in new_cols]
    old_n = len(old_cols)

    for attr in ["mean_", "scale_", "var_"]:
        if hasattr(new_scaler, attr):
            arr = np.asarray(getattr(new_scaler, attr))

            if arr.ndim == 1 and arr.shape[0] == old_n:
                setattr(new_scaler, attr, arr[idx])

    if hasattr(new_scaler, "n_features_in_"):
        new_scaler.n_features_in_ = len(new_cols)

    if hasattr(new_scaler, "feature_names_in_"):
        new_scaler.feature_names_in_ = np.asarray(new_cols, dtype=object)

    return new_scaler


def force_common_gate_object(gate_obj, alpha=None, beta=None):
    if gate_obj is None:
        return None

    gate = deepcopy(gate_obj)

    old_cols = list(gate.get("gate_cont_cols", COMMON_GATE_COLS))
    new_cols = COMMON_GATE_COLS.copy()

    missing = [c for c in new_cols if c not in old_cols]

    if missing:
        raise ValueError(
            "El objeto gate no contiene las variables comunes requeridas. "
            f"Faltan: {missing}. Columnas actuales: {old_cols}"
        )

    if old_cols != new_cols:
        gate["scaler"] = _subset_standard_scaler(
            gate["scaler"],
            old_cols=old_cols,
            new_cols=new_cols,
        )

        if "beta" in gate and beta is None:
            gate["beta"] = float(gate["beta"]) * np.sqrt(len(new_cols) / len(old_cols))

    gate["gate_cont_cols"] = new_cols
    gate["original_gate_cont_cols"] = old_cols
    gate["uses_common_gate_only"] = True

    if alpha is not None:
        gate["alpha"] = float(alpha)

    if beta is not None:
        gate["beta"] = float(beta)
        gate["beta_manual"] = True

    return gate


def gate_param(artifact, name, default):
    gate = artifact.get("gate", {}) or {}

    try:
        gate = force_common_gate_object(gate)
        return float(gate.get(name, default))
    except Exception:
        return float(default)


def apply_common_gate_from_physical(physical_df, gate_obj, periods):
    gate_obj = force_common_gate_object(gate_obj)

    gate_cols = gate_obj["gate_cont_cols"]

    missing_cols = [c for c in gate_cols if c not in physical_df.columns]

    if missing_cols:
        raise ValueError(
            "Faltan columnas físicas para calcular el gate común. "
            f"Faltan: {missing_cols}. Columnas disponibles: {list(physical_df.columns)}"
        )

    X_gate = physical_df[gate_cols].to_numpy(dtype=float)
    X_gate_std = gate_obj["scaler"].transform(X_gate)

    d = np.linalg.norm(X_gate_std, axis=1)

    alpha = float(gate_obj.get("alpha", DEFAULT_ALPHA))
    beta = float(gate_obj.get("beta", DEFAULT_BETA))

    g_dist = 1.0 / (1.0 + np.exp(alpha * (d - beta)))

    if SOIL_COL in physical_df.columns:
        soil_conf = (
            physical_df[SOIL_COL]
            .map(gate_obj.get("soil_conf_map", {}))
            .fillna(DEFAULT_SOIL_CONF)
            .to_numpy(dtype=float)
        )
    else:
        soil_conf = np.full(shape=len(physical_df), fill_value=DEFAULT_SOIL_CONF, dtype=float)

    g_scalar = np.clip(g_dist * soil_conf, 0.0, 1.0)
    G = np.repeat(g_scalar[:, None], repeats=len(periods), axis=1).astype(np.float32)

    return G, g_scalar, g_dist, soil_conf, d


# =========================================================
# CURVAS DE MÉTRICAS
# =========================================================

def build_eval_curves_fallback(artifact, metric):
    try:
        from model.train_model import compute_comparison_curves
    except Exception:
        return None

    Y = np.asarray(artifact["Y"], dtype=float)
    W = np.asarray(artifact["W"], dtype=float)

    meta = artifact.get("df_wide_metadata")

    if meta is None:
        meta = artifact.get("df_wide_preview", pd.DataFrame())

        if len(meta) != W.shape[0]:
            return None

    curves = compute_comparison_curves(
        periods=np.asarray(artifact["periods"], dtype=float),
        Y=Y,
        W=W,
        Yhat_oof=artifact.get("Yhat_oof"),
        Yhat_full=artifact.get("Yhat_full"),
        df_wide=meta,
        fold_summary=artifact.get("fold_summary"),
        min_n_valid_filter=int(artifact.get("min_n_valid_filter", 20)),
    )

    return curves.get(metric)


def get_metric_df(artifact, metric):
    curves = artifact.get("evaluation_curves", {})
    df = curves.get(metric) if isinstance(curves, dict) else None

    if df is None:
        df = build_eval_curves_fallback(artifact, metric)

    return df


def metric_comparison_figure(artifact, metric):
    df = get_metric_df(artifact, metric)

    if df is None or len(df) == 0:
        return empty_figure(
            "No hay curvas comparativas. Vuelve a entrenar con train_model.py actualizado."
        )

    min_n = int(artifact.get("min_n_valid_filter", 20))

    fig = go.Figure()

    traces = [
        (f"{metric}_base", "GMPE base", COLORS["base"], "solid"),
        (f"{metric}_model_oof_all", "GRU OOF — todos los folds", COLORS["oof"], "solid"),
        (
            f"{metric}_model_oof_filtered_nva{min_n}",
            f"GRU OOF — n_valid ≥ {min_n}",
            COLORS["filtered"],
            "dash",
        ),
        (f"{metric}_model_full_train", "GRU full train", COLORS["full"], "dot"),
    ]

    for col, label, color, dash in traces:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Period"],
                    y=df[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2.6, dash=dash),
                    marker=dict(size=6, color=color),
                )
            )

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(title=f"{metric} de residuales logarítmicos")

    return apply_common_layout(
        fig,
        title=f"{metric}: residual base vs OOF vs OOF filtrado vs full train",
        height=480,
    )


def metric_reduction_figure(artifact, metric):
    df = get_metric_df(artifact, metric)

    if df is None or len(df) == 0:
        return empty_figure("No hay reducciones porcentuales disponibles.")

    min_n = int(artifact.get("min_n_valid_filter", 20))

    fig = go.Figure()

    traces = [
        (f"{metric}_reduction_pct_oof_all", "OOF — todos", COLORS["oof"], "solid"),
        (
            f"{metric}_reduction_pct_oof_filtered_nva{min_n}",
            f"OOF — n_valid ≥ {min_n}",
            COLORS["filtered"],
            "dash",
        ),
        (f"{metric}_reduction_pct_full_train", "Full train", COLORS["full"], "dot"),
    ]

    for col, label, color, dash in traces:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Period"],
                    y=df[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2.6, dash=dash),
                    marker=dict(size=6, color=color),
                )
            )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["reference"],
        annotation_text="0%",
        annotation_position="top left",
    )

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(title=f"Reducción relativa de {metric} (%)")

    return apply_common_layout(
        fig,
        title=f"Reducción porcentual de {metric} frente a la GMPE base",
        height=480,
    )


# =========================================================
# RESIDUALES
# =========================================================

def residual_matrix(artifact, mode):
    Y = np.asarray(artifact["Y"], dtype=float)
    W = np.asarray(artifact["W"], dtype=float)

    if mode == "base":
        return np.where(W > 0, Y, np.nan), "Residual base GMPE"

    if mode == "oof":
        yhat = artifact.get("Yhat_oof")

        if yhat is None:
            return None, "OOF no disponible"

        return np.where(W > 0, Y - np.asarray(yhat, dtype=float), np.nan), "Residual corregido OOF"

    if mode == "full":
        yhat = artifact.get("Yhat_full")

        if yhat is None:
            return None, "Full train no disponible"

        return np.where(W > 0, Y - np.asarray(yhat, dtype=float), np.nan), "Residual corregido full train"

    if mode == "full_gated":
        gate_full = artifact.get("gate_full", {})
        eps = gate_full.get("eps_corr_gated_full")

        if eps is None:
            return None, "Full train con gate no disponible"

        return np.asarray(eps, dtype=float), "Residual corregido full train + gate"

    return None, "Modo no reconocido"


def residual_hist_figure(artifact, period, mode):
    eps, label = residual_matrix(artifact, mode)

    if eps is None:
        return empty_figure(label)

    j = selected_period_index(artifact, period)
    T = float(np.asarray(artifact["periods"], dtype=float)[j])

    vals = eps[:, j]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return empty_figure(f"No hay residuales válidos en T={T:g} s.")

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=45,
            name="Residuales",
            marker=dict(
                color=COLORS["hist"],
                line=dict(color=COLORS["hist_line"], width=0.7),
            ),
            opacity=0.88,
        )
    )

    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="solid",
        line_color=COLORS["zero"],
        annotation_text="0",
        annotation_position="top left",
    )

    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color=COLORS["ols"],
        annotation_text="media",
        annotation_position="top right",
    )

    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        text=(
            f"n = {vals.size}<br>"
            f"media = {mean_val:.3f}<br>"
            f"mediana = {median_val:.3f}<br>"
            f"SD = {sd_val:.3f}"
        ),
        font=dict(size=12, color="#1F2937"),
    )

    fig.update_xaxes(title="Residual logarítmico ln(Sa_obs) − ln(Sa_pred)")
    fig.update_yaxes(title="Frecuencia")

    return apply_common_layout(
        fig,
        title=f"Distribución de residuales — {label} — T={T:g} s",
        height=480,
    )


def residual_scatter_figure(artifact, period, mode="oof"):
    W = np.asarray(artifact["W"], dtype=float)

    periods = np.asarray(artifact["periods"], dtype=float)
    j = selected_period_index(artifact, period)
    T = float(periods[j])

    eps, label = residual_matrix(artifact, mode)

    if eps is None:
        return empty_figure(label)

    eps = np.asarray(eps, dtype=float)

    if mode == "oof":
        yhat = artifact.get("Yhat_oof")
        x_label = "Corrección residual predicha por GRU — OOF"
    elif mode == "full":
        yhat = artifact.get("Yhat_full")
        x_label = "Corrección residual predicha por GRU — full train"
    else:
        yhat = None
        x_label = "Índice del registro válido"

    mask = (W[:, j] > 0) & np.isfinite(eps[:, j])

    if yhat is not None:
        yhat = np.asarray(yhat, dtype=float)
        mask = mask & np.isfinite(yhat[:, j])
        x = yhat[mask, j]
    else:
        x = np.arange(mask.sum())

    y = eps[mask, j]

    if len(y) < 3:
        return empty_figure(f"No hay suficientes residuales válidos en T={T:g} s.")

    mean_y = float(np.mean(y))
    median_y = float(np.median(y))
    sd_y = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Residual remanente",
            marker=dict(
                size=7,
                color=COLORS["scatter"],
                opacity=0.62,
                line=dict(color="white", width=0.4),
            ),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>"
                "Residual remanente: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_color=COLORS["zero"],
        line_width=2,
        annotation_text="residual = 0",
        annotation_position="top left",
    )

    fig.add_hline(
        y=mean_y,
        line_color=COLORS["ols"],
        line_width=2,
        line_dash="dash",
        annotation_text="media residual",
        annotation_position="bottom left",
    )

    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        text=(
            f"n = {len(y)}<br>"
            f"media = {mean_y:.3f}<br>"
            f"mediana = {median_y:.3f}<br>"
            f"SD = {sd_y:.3f}"
        ),
        font=dict(size=12, color="#1F2937"),
    )

    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Residual remanente logarítmico")

    return apply_common_layout(
        fig,
        title=f"Diagnóstico de residuales — {label} — T={T:g} s",
        height=500,
    )


# =========================================================
# VISUALIZACIÓN DEL GATE
# =========================================================

def gate_variable_options(model_key=None):
    return [
        {"label": "Magnitud Mw", "value": "Magnitude"},
        {"label": "Rrup (km)", "value": "Rrup_km"},
        {"label": "Profundidad hipocentral (km)", "value": "Hypocenter Depth (km)"},
    ]


def pretty_axis_label(var):
    labels = {
        "Magnitude": "Magnitud Mw",
        "Rrup_km": "Rrup (km)",
        "Hypocenter Depth (km)": "Profundidad hipocentral (km)",
    }

    return labels.get(var, var)


def normalize_column_name(name):
    text = str(name).strip().lower()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i")
    text = text.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", "_").replace("-", "_")
    text = "_".join(text.split())
    return text


def resolve_physical_column(df, var):
    candidates = {
        "Magnitude": ["Magnitude", "magnitude", "Mw", "mag"],
        "Rrup_km": [
            "Rrup_OpenQuake",
            "Rrup (km)",
            "Rrup",
            "rrup_openquake",
            "rrup_km",
            "Rrup_km",
            "ln_Rrup",
            "log_Rrup",
        ],
        "Hypocenter Depth (km)": [
            "Hypocenter Depth (km)",
            "Hypocentral Depth (km)",
            "Depth",
            "Depth_km",
            "zhypo_km",
            "Zhypo",
        ],
    }

    possible = candidates.get(var, [var])
    normalized = {normalize_column_name(c): c for c in df.columns}

    for cand in possible:
        key = normalize_column_name(cand)
        if key in normalized:
            return normalized[key]

    for cand in possible:
        key = normalize_column_name(cand)
        for norm_col, real_col in normalized.items():
            if key and key in norm_col:
                return real_col

    return None


def force_physical_units(var, values, source_col=None):
    s = pd.to_numeric(values, errors="coerce")
    colname = str(source_col or "").lower()

    if var == "Rrup_km":
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()

        if not finite.empty:
            looks_logged_by_name = ("ln" in colname) or ("log" in colname)
            looks_logged_by_range = finite.quantile(0.95) < 10 and finite.max() <= 10

            if looks_logged_by_name or looks_logged_by_range:
                s = np.exp(s)

    return s


def get_gate_training_points(artifact, xvar, yvar, max_points=GATE_MAX_POINTS):
    sources = []

    candidate_paths = [
        Path(__file__).resolve().parents[1] / "Resids_for_Eliasib.xlsx",
        Path.cwd() / "Resids_for_Eliasib.xlsx",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                raw = pd.read_excel(path)
                sources.append(raw)
                break
            except Exception:
                pass

    for key in ["df_wide_metadata", "df_wide_preview", "metadata", "df_wide"]:
        obj = artifact.get(key)

        if isinstance(obj, pd.DataFrame) and len(obj) > 0:
            sources.append(obj.copy())

    for df in sources:
        xcol = resolve_physical_column(df, xvar)
        ycol = resolve_physical_column(df, yvar)

        if xcol is None or ycol is None:
            continue

        x = force_physical_units(xvar, df[xcol], xcol)
        y = force_physical_units(yvar, df[ycol], ycol)

        out = pd.DataFrame({"x": x, "y": y})
        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        out = out.drop_duplicates()

        if len(out) == 0:
            continue

        if len(out) > max_points:
            out = out.sample(max_points, random_state=42)

        return out

    return pd.DataFrame(columns=["x", "y"])


def gate_contour_figure(artifact, xvar, yvar, magnitude, rrup, zhypo, soil, alpha, beta):
    gate_obj = artifact.get("gate")

    if gate_obj is None:
        return empty_figure(
            "Este model.pkl no contiene objeto gate. Reentrena o desactiva la visualización del gate.",
            height=620,
        )

    if xvar == yvar:
        return empty_figure(
            "Selecciona dos variables distintas para la curva de nivel del gate.",
            height=620,
        )

    try:
        gate_obj = force_common_gate_object(gate_obj, alpha=alpha, beta=beta)
    except Exception as exc:
        return empty_figure(f"No se pudo adaptar el gate común: {exc}", height=620)

    ranges = {
        "Magnitude": np.linspace(4.0, 8.0, GATE_GRID_N),
        "Rrup_km": np.linspace(0.1, 400.0, GATE_GRID_N),
        "Hypocenter Depth (km)": np.linspace(0.0, 80.0, GATE_GRID_N),
    }

    axis_ranges = {
        "Magnitude": [4.0, 8.0],
        "Rrup_km": [0.0, 400.0],
        "Hypocenter Depth (km)": [0.0, 80.0],
    }

    if xvar not in ranges or yvar not in ranges:
        return empty_figure(
            "Variable no reconocida. El gate solo usa Magnitude, Rrup_km y Hypocenter Depth.",
            height=620,
        )

    xs = ranges[xvar]
    ys = ranges[yvar]

    XX, YY = np.meshgrid(xs, ys)

    base = {
        "Magnitude": float(magnitude),
        "Rrup_km": float(rrup),
        "Hypocenter Depth (km)": float(zhypo),
        SOIL_COL: int(soil),
    }

    rows = []

    for xv, yv in zip(XX.ravel(), YY.ravel()):
        row = base.copy()
        row[xvar] = float(xv)
        row[yvar] = float(yv)
        rows.append(row)

    physical_df = pd.DataFrame(rows)

    try:
        _, g_scalar, _, _, _ = apply_common_gate_from_physical(
            physical_df,
            gate_obj,
            np.asarray(artifact["periods"], dtype=float),
        )
    except Exception as exc:
        return empty_figure(f"No se pudo calcular el gate común: {exc}", height=620)

    Z = np.asarray(g_scalar, dtype=float).reshape(XX.shape)
    Z = np.clip(Z, 0.0, 1.0)

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=Z,
            zmin=0,
            zmax=1,
            colorscale=GATE_COLORSCALE,
            opacity=0.96,
            contours=dict(
                start=0,
                end=1,
                size=0.1,
                showlabels=True,
                labelfont=dict(size=10, color="#111827"),
            ),
            colorbar=dict(
                title="w_gate",
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.50", "0.75", "1"],
                len=0.78,
                thickness=20,
            ),
            hovertemplate=(
                f"{pretty_axis_label(xvar)}=%{{x:.3g}}<br>"
                f"{pretty_axis_label(yvar)}=%{{y:.3g}}<br>"
                "w_gate=%{z:.3f}<extra></extra>"
            ),
            name="w_gate",
        )
    )

    points = get_gate_training_points(
        artifact=artifact,
        xvar=xvar,
        yvar=yvar,
        max_points=GATE_MAX_POINTS,
    )

    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["x"],
                y=points["y"],
                mode="markers",
                name="Datos observados",
                marker=dict(
                    size=6,
                    color=COLORS["points"],
                    opacity=0.88,
                    line=dict(color=COLORS["points_border"], width=0.7),
                ),
                hovertemplate=(
                    f"{pretty_axis_label(xvar)}=%{{x:.3g}}<br>"
                    f"{pretty_axis_label(yvar)}=%{{y:.3g}}"
                    "<extra>Dato observado</extra>"
                ),
            )
        )

    if xvar in base and yvar in base:
        fig.add_trace(
            go.Scatter(
                x=[base[xvar]],
                y=[base[yvar]],
                mode="markers",
                name="Escenario fijo",
                marker=dict(
                    color="#DC2626",
                    size=14,
                    symbol="x",
                    line=dict(color="white", width=2.3),
                ),
                hovertemplate=(
                    f"{pretty_axis_label(xvar)}=%{{x:.3g}}<br>"
                    f"{pretty_axis_label(yvar)}=%{{y:.3g}}"
                    "<extra>Escenario fijo</extra>"
                ),
            )
        )

    fig.update_xaxes(
        title=pretty_axis_label(xvar),
        type="linear",
        range=axis_ranges[xvar],
    )

    fig.update_yaxes(
        title=pretty_axis_label(yvar),
        type="linear",
        range=axis_ranges[yvar],
    )

    fig = apply_common_layout(
        fig,
        title="Curvas de nivel del peso del gate",
        height=620,
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=65, r=70, t=75, b=65),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            font=dict(size=11),
        ),
    )

    return fig


def gate_sigmoid_figure(alpha, beta):
    alpha = float(alpha or DEFAULT_ALPHA)
    beta = float(beta or DEFAULT_BETA)

    dmax = max(beta * 2.0, beta + 4.0, 8.0)
    d = np.linspace(0.0, dmax, 400)

    g_dist = 1.0 / (1.0 + np.exp(alpha * (d - beta)))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d,
            y=g_dist,
            mode="lines",
            name="g_dist",
            line=dict(color=COLORS["oof"], width=3),
            hovertemplate="d=%{x:.3f}<br>g_dist=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=beta,
        line_color=COLORS["ols"],
        line_width=2,
        line_dash="dash",
        annotation_text=f"β = {beta:.3f}",
        annotation_position="top right",
    )

    fig.add_hline(
        y=0.5,
        line_color=COLORS["reference"],
        line_width=1.8,
        line_dash="dot",
        annotation_text="0.5",
        annotation_position="bottom left",
    )

    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="#CBD5E1",
        borderwidth=1,
        text=(
            f"α = {alpha:.3f}<br>"
            f"β = {beta:.3f}<br>"
            "g(d)=1/[1+exp(α(d−β))]"
        ),
        font=dict(size=12, color="#1F2937"),
    )

    fig.update_xaxes(title="Distancia estandarizada al dominio de entrenamiento, d")
    fig.update_yaxes(title="Peso por distancia, g_dist", range=[-0.03, 1.03])

    fig = apply_common_layout(
        fig,
        title="Función sigmoide del gate",
        height=620,
    )

    fig.update_layout(
        margin=dict(l=65, r=35, t=75, b=65),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            font=dict(size=11),
        ),
    )

    return fig


# =========================================================
# LAYOUT
# =========================================================

def layout():
    package = get_package()

    if isinstance(package, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("Aún no hay model/model.pkl disponible.", className="alert-heading"),
                    html.P(str(package)),
                    html.P(
                        "Ejecuta: python model/train_model.py --models nosam,nosam_elevation,ask14 --epochs 60"
                    ),
                ],
                color="warning",
            ),
            fluid=True,
        )

    opts = model_options(package)

    if not opts:
        return dbc.Container(
            dbc.Alert(
                "model.pkl existe, pero no contiene modelos en package['models'].",
                color="danger",
            ),
            fluid=True,
        )

    default_model = opts[0]["value"]
    default_artifact = package["models"][default_model]

    p_opts = period_options(default_artifact)
    default_period = p_opts[0]["value"] if p_opts else None
    default_alpha = DEFAULT_ALPHA
    default_beta = DEFAULT_BETA

    return dbc.Container(
        [
            html.Div("Evaluación del modelo", className="section-kicker"),
            html.H1("Resultados: LOGO, full train, reducciones, gate y variabilidad", className="page-title"),
            html.P(
                "Las curvas comparan el residual base de la GMPE contra el residual corregido por la GRU: "
                "OOF con todos los folds, OOF filtrado por eventos con al menos 20 registros de validación "
                "y entrenamiento full. También se incluye la descomposición de variabilidad τ, ϕ y σ "
                "entre NoSAm base y ASK14 residual corregido.",
                className="lead",
            ),

            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Modelo"),
                                    dcc.Dropdown(
                                        id="results-model-dropdown",
                                        options=opts,
                                        value=default_model,
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Métrica"),
                                    dcc.Dropdown(
                                        id="results-metric",
                                        options=METRIC_OPTIONS,
                                        value="SD",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Período para residuales"),
                                    dcc.Dropdown(
                                        id="results-period",
                                        options=p_opts,
                                        value=default_period,
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Tipo de residual"),
                                    dcc.Dropdown(
                                        id="results-residual-mode",
                                        options=[
                                            {"label": "Base GMPE", "value": "base"},
                                            {"label": "Corregido OOF", "value": "oof"},
                                            {"label": "Corregido full train", "value": "full"},
                                            {
                                                "label": "Corregido full train + gate",
                                                "value": "full_gated",
                                            },
                                        ],
                                        value="oof",
                                        clearable=False,
                                    ),
                                ],
                                md=2,
                            ),
                        ],
                        className="g-3",
                    )
                ),
                className="soft-card mb-4",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="fig-metric-comparison")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="fig-metric-reduction")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="fig-scatter")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="fig-residual-hist")),
                            className="soft-card",
                        ),
                        lg=6,
                    ),
                ],
                className="g-4 mb-4",
            ),

            decomposition_card(
                prefix="resdecomp",
                title="Descomposición de variabilidad: NoSAm base vs ASK14 residual corregido",
                use_external_gate_controls=False,
            ),

            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Curvas de nivel y función del gate", className="mb-3"),
                        dbc.Alert(
                            "El gate se calcula fuera de la GRU, de forma post hoc. "
                            "En esta pestaña se fuerza a usar solo Magnitude, Rrup y Hypocenter Depth para que "
                            "NoSAm, NoSAm + Elevation y ASK14 sean comparables. Rvolc y Station Elevation no entran al gate.",
                            color="light",
                            className="border mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Eje X"),
                                        dcc.Dropdown(
                                            id="gate-xvar",
                                            value="Magnitude",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Eje Y"),
                                        dcc.Dropdown(
                                            id="gate-yvar",
                                            value="Rrup_km",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Alpha"),
                                        dbc.Input(
                                            id="gate-alpha",
                                            type="number",
                                            value=DEFAULT_ALPHA,
                                            step=0.0001,
                                            disabled=True,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Beta"),
                                        dbc.Input(
                                            id="gate-beta",
                                            type="number",
                                            value=DEFAULT_BETA,
                                            step=0.0001,
                                            disabled=True,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Mw fijo"),
                                        dbc.Input(
                                            id="gate-mw",
                                            type="number",
                                            value=5.5,
                                            step=0.1,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Rrup fijo"),
                                        dbc.Input(
                                            id="gate-rrup",
                                            type="number",
                                            value=100.0,
                                            step=5,
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Zhypo fijo"),
                                        dbc.Input(
                                            id="gate-zhypo",
                                            type="number",
                                            value=10.0,
                                            step=1,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Suelo"),
                                        dbc.Input(
                                            id="gate-soil",
                                            type="number",
                                            value=2,
                                            min=1,
                                            max=5,
                                            step=1,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Actualizar"),
                                        dbc.Button(
                                            "Actualizar gate",
                                            id="gate-update-button",
                                            color="dark",
                                            className="w-100",
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-3",
                        ),
                    ]
                ),
                className="soft-card mb-4",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(
                                    id="fig-gate-contour",
                                    style={"height": "650px"},
                                    config={"responsive": True},
                                )
                            ),
                            className="soft-card h-100",
                        ),
                        lg=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(
                                    id="fig-gate-sigmoid",
                                    style={"height": "650px"},
                                    config={"responsive": True},
                                )
                            ),
                            className="soft-card h-100",
                        ),
                        lg=6,
                    ),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )


# =========================================================
# CALLBACKS
# =========================================================

def register_callbacks(app):
    @app.callback(
        Output("results-period", "options"),
        Output("results-period", "value"),
        Output("gate-xvar", "options"),
        Output("gate-yvar", "options"),
        Output("gate-alpha", "value"),
        Output("gate-beta", "value"),
        Input("results-model-dropdown", "value"),
    )
    def update_model_dependent_options(model_key):
        package = get_package()

        if isinstance(package, Exception) or model_key not in package.get("models", {}):
            return [], None, [], [], DEFAULT_ALPHA, DEFAULT_BETA

        artifact = package["models"][model_key]
        p_opts = period_options(artifact)
        g_opts = gate_variable_options(model_key)

        return (
            p_opts,
            p_opts[0]["value"] if p_opts else None,
            g_opts,
            g_opts,
            DEFAULT_ALPHA,
            DEFAULT_BETA,
        )

    @app.callback(
        Output("fig-metric-comparison", "figure"),
        Output("fig-metric-reduction", "figure"),
        Output("fig-scatter", "figure"),
        Output("fig-residual-hist", "figure"),
        Input("results-model-dropdown", "value"),
        Input("results-metric", "value"),
        Input("results-period", "value"),
        Input("results-residual-mode", "value"),
    )
    def update_metric_and_residual_figures(
        model_key,
        metric,
        period,
        residual_mode,
    ):
        package = get_package()

        if isinstance(package, Exception) or model_key not in package.get("models", {}):
            fig = empty_figure("Modelo no disponible.")
            return fig, fig, fig, fig

        artifact = package["models"][model_key]

        return (
            metric_comparison_figure(artifact, metric),
            metric_reduction_figure(artifact, metric),
            residual_scatter_figure(artifact, period, residual_mode),
            residual_hist_figure(artifact, period, residual_mode),
        )

    @app.callback(
        Output("fig-gate-contour", "figure"),
        Output("fig-gate-sigmoid", "figure"),
        Input("gate-update-button", "n_clicks"),
        Input("results-model-dropdown", "value"),
        State("gate-xvar", "value"),
        State("gate-yvar", "value"),
        State("gate-mw", "value"),
        State("gate-rrup", "value"),
        State("gate-zhypo", "value"),
        State("gate-soil", "value"),
        State("gate-alpha", "value"),
        State("gate-beta", "value"),
    )
    def update_gate_figures(
        n_clicks,
        model_key,
        xvar,
        yvar,
        mw,
        rrup,
        zhypo,
        soil,
        alpha,
        beta,
    ):
        package = get_package()

        if isinstance(package, Exception) or model_key not in package.get("models", {}):
            fig = empty_figure("Modelo no disponible.", height=620)
            return fig, fig

        artifact = package["models"][model_key]

        alpha = DEFAULT_ALPHA if alpha in [None, ""] else float(alpha)
        beta = DEFAULT_BETA if beta in [None, ""] else float(beta)

        return (
            gate_contour_figure(
                artifact=artifact,
                xvar=xvar,
                yvar=yvar,
                magnitude=float(mw or 5.5),
                rrup=float(rrup or 100),
                zhypo=float(zhypo or 10),
                soil=int(soil or 2),
                alpha=alpha,
                beta=beta,
            ),
            gate_sigmoid_figure(
                alpha=alpha,
                beta=beta,
            ),
        )

    register_decomposition_callbacks(
        app,
        prefix="resdecomp",
        use_external_gate_controls=False,
    )