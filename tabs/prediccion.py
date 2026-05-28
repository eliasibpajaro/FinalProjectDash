from copy import deepcopy
from functools import lru_cache

from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURACIÓN VISUAL
# =========================================================

MODEL_STYLE = {
    "nosam": {
        "color": "#1F2937",
        "soft": "rgba(31, 41, 55, 0.22)",
        "label": "NoSAm",
    },
    "nosam_elevation": {
        "color": "#0F766E",
        "soft": "rgba(15, 118, 110, 0.22)",
        "label": "NoSAm + Elevation",
    },
    "ask14": {
        "color": "#1D4ED8",
        "soft": "rgba(29, 78, 216, 0.22)",
        "label": "ASK14",
    },
}

NEUTRAL = {
    "text": "#111827",
    "muted": "#64748B",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "observed": "rgba(71, 85, 105, 0.52)",
    "gate": "#B45309",
    "sigma_base": "rgba(31, 41, 55, 0.12)",
}

SENSITIVITY_COLORS = [
    "#1F2937",
    "#334155",
    "#0F766E",
    "#1D4ED8",
    "#7C3AED",
    "#B45309",
    "#9F1239",
    "#475569",
]

COMMON_GATE_COLS = ["Rrup_km", "Magnitude", "Hypocenter Depth (km)"]

VARIABLE_OPTIONS_BASE = [
    {"label": "Magnitud Mw", "value": "magnitude"},
    {"label": "Rrup (km)", "value": "rrup_km"},
    {"label": "Profundidad hipocentral (km)", "value": "zhypo_km"},
    {"label": "Clase de suelo", "value": "soil_class"},
]

_PACKAGE_CACHE = None
_KERAS_MODEL_CACHE = {}


def model_color(model_key):
    return MODEL_STYLE.get(model_key, MODEL_STYLE["nosam"])["color"]


def model_soft_color(model_key):
    return MODEL_STYLE.get(model_key, MODEL_STYLE["nosam"])["soft"]


def apply_prediction_layout(fig, title, height=640, legend_y=-0.22):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=45, r=35, t=70, b=95),
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            size=13,
            color=NEUTRAL["text"],
        ),
        title_font=dict(size=18, color=NEUTRAL["text"]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            y=legend_y,
            x=0,
            font=dict(size=12),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=NEUTRAL["grid"],
        zeroline=False,
        linecolor=NEUTRAL["axis"],
        tickfont=dict(color="#374151"),
        title_font=dict(color=NEUTRAL["text"]),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=NEUTRAL["grid"],
        zeroline=False,
        linecolor=NEUTRAL["axis"],
        tickfont=dict(color="#374151"),
        title_font=dict(color=NEUTRAL["text"]),
    )

    return fig


def empty_figure(message, height=560):
    fig = go.Figure()
    fig.add_annotation(
        text=str(message),
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=NEUTRAL["muted"]),
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
# CARGA DE MODELO
# =========================================================

def get_package(force_reload=False):
    global _PACKAGE_CACHE

    if _PACKAGE_CACHE is not None and not force_reload:
        return _PACKAGE_CACHE

    try:
        from model.train_model import load_model_package

        _PACKAGE_CACHE = load_model_package("model/model.pkl")
        return _PACKAGE_CACHE

    except Exception as exc:
        return exc


def get_cached_keras_model(model_key, artifact):
    if model_key not in _KERAS_MODEL_CACHE:
        from model.train_model import unpack_keras_model

        _KERAS_MODEL_CACHE[model_key] = unpack_keras_model(artifact["model"])

    return _KERAS_MODEL_CACHE[model_key]


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


def variable_options(model_key):
    opts = VARIABLE_OPTIONS_BASE.copy()

    if model_key in ["nosam", "nosam_elevation"]:
        opts.append({"label": "Rvolc (km)", "value": "rvolc_km"})

    if model_key == "nosam_elevation":
        opts.append({"label": "Station Elevation (m)", "value": "station_elevation_m"})

    return opts


# =========================================================
# UTILIDADES
# =========================================================

def parse_number(value, default):
    if value is None:
        return float(default)

    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return float(default)

    return float(value)


def parse_optional_number(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None

    return float(value)


def fmt_number(value):
    if value is None:
        return "NA"

    try:
        return f"{float(value):.3g}"
    except Exception:
        return str(value)


def scenario_dict(magnitude, rrup, zhypo, soil, rvolc, elev):
    return {
        "magnitude": parse_number(magnitude, 5.5),
        "rrup_km": parse_number(rrup, 100.0),
        "zhypo_km": parse_number(zhypo, 10.0),
        "soil_class": int(parse_number(soil, 2)),
        "rvolc_km": parse_number(rvolc, 0.0),
        "station_elevation_m": parse_number(elev, 100.0),
    }


# =========================================================
# GATE COMÚN
# =========================================================

def _subset_standard_scaler(scaler, old_cols, new_cols):
    """
    Recorta un StandardScaler ya entrenado para ignorar columnas que no deben
    entrar al gate. Esto permite usar un model.pkl anterior sin reentrenar.
    """
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


def force_common_gate_object(gate_obj, alpha_value=None, beta_value=None):
    """
    Adapta un gate viejo o nuevo para que use solo las variables comunes:
    Rrup_km, Magnitude y Hypocenter Depth (km).

    Esto NO cambia la GMPE ni la GRU. Rvolc y Station Elevation pueden seguir
    entrando al modelo cuando corresponda, pero no entran al gate.
    """
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

        if "beta" in gate and beta_value is None:
            gate["beta"] = float(gate["beta"]) * np.sqrt(len(new_cols) / len(old_cols))

    gate["gate_cont_cols"] = new_cols
    gate["original_gate_cont_cols"] = old_cols
    gate["uses_common_gate_only"] = True

    alpha = parse_optional_number(alpha_value)
    beta = parse_optional_number(beta_value)

    if alpha is not None:
        if alpha <= 0:
            raise ValueError("alpha debe ser mayor que 0.")
        gate["alpha"] = float(alpha)

    if beta is not None:
        if beta <= 0:
            raise ValueError("beta debe ser mayor que 0.")
        gate["beta"] = float(beta)
        gate["beta_manual"] = True

    return gate


def make_gate_object(artifact, apply_gate, alpha_value=None, beta_value=None):
    if not apply_gate:
        return None

    if "gate" not in artifact or artifact["gate"] is None:
        raise ValueError(
            "Este model.pkl no contiene objeto gate. "
            "Desactiva el gate o reentrena."
        )

    return force_common_gate_object(
        artifact["gate"],
        alpha_value=alpha_value,
        beta_value=beta_value,
    )


def common_gate_default_params(artifact):
    gate = artifact.get("gate", {}) or {}

    try:
        gate = force_common_gate_object(gate)
        return gate.get("alpha", None), gate.get("beta", None)
    except Exception:
        return gate.get("alpha", None), gate.get("beta", None)


def gate_info_text(gate_obj, apply_gate):
    if not apply_gate:
        return "Gate aplicado: no | corrección GRU completa"

    if gate_obj is None:
        return "Gate aplicado: no disponible"

    original = gate_obj.get("original_gate_cont_cols")
    original_txt = ""

    if original and original != gate_obj.get("gate_cont_cols"):
        original_txt = f" | gate original={original}"

    return (
        "Gate aplicado: sí | "
        f"variables usadas={gate_obj.get('gate_cont_cols')} | "
        f"alpha={fmt_number(gate_obj.get('alpha'))} | "
        f"beta={fmt_number(gate_obj.get('beta'))}"
        f"{original_txt}"
    )


# =========================================================
# PREDICCIÓN ESPECTRAL
# =========================================================

def predict_many(
    package,
    model_key,
    scenarios,
    apply_gate=True,
    gate_alpha=None,
    gate_beta=None,
):
    try:
        from model.train_model import (
            build_prediction_frame,
            build_sequence_input,
            base_sa_spectrum,
            gate_weight_single,
        )
    except Exception as exc:
        raise ImportError(
            "No se pudieron importar funciones desde model.train_model: "
            "build_prediction_frame, build_sequence_input, base_sa_spectrum y gate_weight_single. "
            f"Error original: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(package, dict) or "models" not in package:
        raise ValueError("El model.pkl no tiene la estructura esperada: falta package['models'].")

    if model_key not in package["models"]:
        raise ValueError(
            f"Modelo no disponible: {model_key}. "
            f"Disponibles: {list(package['models'].keys())}"
        )

    artifact = package["models"][model_key]

    required = ["periods", "model", "preprocessor", "feature_cols"]
    missing = [k for k in required if k not in artifact]

    if missing:
        raise ValueError(f"El artefacto '{model_key}' no contiene estas claves: {missing}.")

    periods = np.asarray(artifact["periods"], dtype=float)
    model = get_cached_keras_model(model_key, artifact)
    preprocessor = artifact["preprocessor"]

    gate_obj = make_gate_object(
        artifact,
        apply_gate,
        alpha_value=gate_alpha,
        beta_value=gate_beta,
    )

    frames = []

    for sc in scenarios:
        frames.append(
            build_prediction_frame(
                model_key=model_key,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
                station_elevation_m=sc.get("station_elevation_m", 100.0),
            )
        )

    X_df = pd.concat(frames, ignore_index=True)

    missing_features = [c for c in artifact["feature_cols"] if c not in X_df.columns]

    if missing_features:
        raise ValueError(f"Faltan columnas para predecir con '{model_key}': {missing_features}")

    Xp = preprocessor.transform(X_df[artifact["feature_cols"]])

    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    Xp = np.asarray(Xp, dtype=np.float32)

    X_seq = build_sequence_input(Xp, periods)

    residual_hat = np.asarray(model.predict(X_seq, verbose=0), dtype=float)

    if residual_hat.ndim == 3 and residual_hat.shape[-1] == 1:
        residual_hat = residual_hat[:, :, 0]

    if residual_hat.ndim == 1:
        residual_hat = residual_hat.reshape(1, -1)

    base_list = []
    corrected_list = []
    gate_list = []
    base_name = None

    for i, sc in enumerate(scenarios):
        base_sa, base_name_i = base_sa_spectrum(
            model_key=model_key,
            periods=periods,
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        base_name = base_name_i

        if apply_gate:
            w_gate = gate_weight_single(
                gate_obj=gate_obj,
                model_key=model_key,
                periods=periods,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
            )
        else:
            w_gate = 1.0

        base_sa = np.asarray(base_sa, dtype=float)
        corrected_sa = base_sa * np.exp(residual_hat[i, :] * float(w_gate))

        base_list.append(base_sa)
        corrected_list.append(corrected_sa)
        gate_list.append(float(w_gate))

    return {
        "artifact": artifact,
        "periods": periods,
        "residual_hat": residual_hat,
        "base_sa": np.asarray(base_list, dtype=float),
        "corrected_sa": np.asarray(corrected_list, dtype=float),
        "gate_weight": np.asarray(gate_list, dtype=float),
        "base_name": base_name,
        "apply_gate": bool(apply_gate),
        "gate_obj": gate_obj,
    }


# =========================================================
# ATENUACIÓN
# =========================================================

def predict_attenuation_curve(
    package,
    model_key,
    base_scenario,
    selected_period,
    apply_gate=True,
    gate_alpha=None,
    gate_beta=None,
    n_points=35,
):
    from model.train_model import (
        build_prediction_frame,
        build_sequence_input,
        base_sa_spectrum,
        gate_weight_single,
    )

    artifact = package["models"][model_key]
    periods = np.asarray(artifact["periods"], dtype=float)

    j = int(np.argmin(np.abs(periods - float(selected_period))))
    T = float(periods[j])

    model = get_cached_keras_model(model_key, artifact)
    preprocessor = artifact["preprocessor"]

    gate_obj = make_gate_object(
        artifact,
        apply_gate,
        alpha_value=gate_alpha,
        beta_value=gate_beta,
    )

    n_points = max(10, min(int(n_points or 35), 120))
    rrup_grid = np.geomspace(5.0, 400.0, n_points)

    scenarios = []

    for rr in rrup_grid:
        sc = base_scenario.copy()
        sc["rrup_km"] = float(rr)
        scenarios.append(sc)

    frames = []

    for sc in scenarios:
        frames.append(
            build_prediction_frame(
                model_key=model_key,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
                station_elevation_m=sc.get("station_elevation_m", 100.0),
            )
        )

    X_df = pd.concat(frames, ignore_index=True)

    Xp = preprocessor.transform(X_df[artifact["feature_cols"]])

    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    Xp = np.asarray(Xp, dtype=np.float32)

    residual_hat = np.asarray(
        model.predict(build_sequence_input(Xp, periods), verbose=0),
        dtype=float,
    )

    if residual_hat.ndim == 3 and residual_hat.shape[-1] == 1:
        residual_hat = residual_hat[:, :, 0]

    residual_T = residual_hat[:, j]

    base_sa_T = []
    corrected_sa_T = []
    gate_weights = []
    base_name = None

    for i, sc in enumerate(scenarios):
        base_sa_one, base_name_i = base_sa_spectrum(
            model_key=model_key,
            periods=np.array([T], dtype=float),
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        base_name = base_name_i
        base_val = float(np.asarray(base_sa_one).reshape(-1)[0])

        if apply_gate:
            w_gate = gate_weight_single(
                gate_obj=gate_obj,
                model_key=model_key,
                periods=periods,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
            )
        else:
            w_gate = 1.0

        base_sa_T.append(base_val)
        corrected_sa_T.append(base_val * np.exp(float(w_gate) * float(residual_T[i])))
        gate_weights.append(float(w_gate))

    return {
        "T": T,
        "rrup_grid": rrup_grid,
        "base_sa_T": np.asarray(base_sa_T, dtype=float),
        "corrected_sa_T": np.asarray(corrected_sa_T, dtype=float),
        "gate_weight": np.asarray(gate_weights, dtype=float),
        "base_name": base_name,
        "gate_obj": gate_obj,
    }


# =========================================================
# FIGURAS
# =========================================================

def spectrum_single_figure(result, model_key):
    periods = result["periods"]
    color = model_color(model_key)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=periods,
            y=result["base_sa"][0],
            mode="lines+markers",
            name=f"{result['base_name']} base",
            line=dict(color=color, width=2.8, dash="dash"),
            marker=dict(
                size=6,
                color=color,
                symbol="circle-open",
                line=dict(width=1.8),
            ),
            legendgroup=model_key,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=periods,
            y=result["corrected_sa"][0],
            mode="lines+markers",
            name=f"{result['base_name']} + GRU corregido",
            line=dict(color=color, width=3.3),
            marker=dict(size=6, color=color),
            legendgroup=model_key,
        )
    )

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(type="log", title="Sa (g)")

    return apply_prediction_layout(
        fig,
        "Espectro base vs espectro corregido",
        height=640,
        legend_y=-0.20,
    )


def sensitivity_values(var_name, vmin, vmax, n):
    n = max(2, min(int(n or 4), 8))

    vmin = parse_number(vmin, 0.0)
    vmax = parse_number(vmax, 1.0)

    if vmax <= vmin:
        vmax = vmin + 1.0

    if var_name == "soil_class":
        vals = [int(round(v)) for v in np.linspace(vmin, vmax, n)]
        vals = [min(max(v, 1), 5) for v in vals]
        return sorted(list(dict.fromkeys(vals)))

    if var_name == "rrup_km":
        return np.geomspace(max(vmin, 0.1), max(vmax, 0.2), n)

    return np.linspace(vmin, vmax, n)


def sensitivity_figure(result, varied_variable, values, model_key):
    periods = result["periods"]
    fig = go.Figure()

    for i, val in enumerate(values):
        color = SENSITIVITY_COLORS[i % len(SENSITIVITY_COLORS)]
        label_val = f"{varied_variable}={float(val):g}"

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=result["base_sa"][i],
                mode="lines",
                name=f"Base | {label_val}",
                line=dict(color=color, width=2.2, dash="dash"),
                legendgroup=f"scenario-{i}",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=result["corrected_sa"][i],
                mode="lines",
                name=f"Corregido | {label_val} | w={result['gate_weight'][i]:.2f}",
                line=dict(color=color, width=3.0),
                legendgroup=f"scenario-{i}",
            )
        )

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(type="log", title="Sa (g)")

    return apply_prediction_layout(
        fig,
        "Sensibilidad espectral",
        height=700,
        legend_y=-0.32,
    )


@lru_cache(maxsize=1)
def load_observed_database_cached():
    from model.train_model import (
        read_table,
        standardize_columns,
        PERIOD_COL,
        TMAX_COL,
        SA_COL,
        RVOLC_COL,
    )

    df = standardize_columns(read_table("Resids_for_Eliasib.xlsx"))

    for col in [
        PERIOD_COL,
        TMAX_COL,
        SA_COL,
        "Magnitude",
        "Hypocenter Depth (km)",
        "Rrup_OpenQuake",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if RVOLC_COL in df.columns:
        df[RVOLC_COL] = pd.to_numeric(df[RVOLC_COL], errors="coerce")

    return df


def load_observed_for_attenuation(model_key, selected_period, mw, zhypo, soil, rvolc):
    try:
        from model.train_model import PERIOD_COL, TMAX_COL, SA_COL, SOIL_COL, RVOLC_COL

        df = load_observed_database_cached().copy()

        dff = df[
            np.isclose(
                df[PERIOD_COL].astype(float),
                float(selected_period),
                atol=1e-10,
                rtol=0,
            )
        ].copy()

        dff = dff[dff[PERIOD_COL] <= dff[TMAX_COL]]
        dff = dff[dff["Magnitude"].between(float(mw) - 0.20, float(mw) + 0.20)]
        dff = dff[dff["Hypocenter Depth (km)"].between(float(zhypo) - 5.0, float(zhypo) + 5.0)]
        dff = dff[dff[SOIL_COL].astype(float).round().astype(int) == int(soil)]

        if model_key in ["nosam", "nosam_elevation"] and RVOLC_COL in dff.columns:
            dff = dff[dff[RVOLC_COL].between(float(rvolc) - 20.0, float(rvolc) + 20.0)]

        dff["Rrup_km"] = dff["Rrup_OpenQuake"]
        dff = dff[np.isfinite(dff["Rrup_km"]) & np.isfinite(dff[SA_COL])]

        return dff[["Rrup_km", SA_COL]].rename(columns={SA_COL: "Sa"}).copy()

    except Exception:
        return pd.DataFrame(columns=["Rrup_km", "Sa"])


def find_sigma_for_period(artifact, selected_period, curve_type="corrected"):
    """
    Busca la desviación estándar correspondiente a un período.

    Para la banda base usa preferentemente:
    - SD_base

    Para la banda corregida usa preferentemente:
    - SD_model_full_train
    - SD_model_oof_filtered_nva20
    - SD_model_oof_all

    Esto corrige el problema de que la banda corregida no aparecía porque
    el nombre real de la columna no estaba incluido en la búsqueda.
    """
    periods = np.asarray(artifact.get("periods", []), dtype=float)

    if periods.size == 0:
        return None

    selected_period = float(selected_period)
    j = int(np.argmin(np.abs(periods - selected_period)))

    if curve_type == "base":
        preferred_names = [
            "SD_base",
            "sd_base",
            "SD_base_full",
            "sd_base_full",
            "sigma_base",
            "gmpe_sigma",
            "sd_gmpe",
        ]
    else:
        preferred_names = [
            "SD_model_full_train",
            "sd_model_full_train",
            "SD_model_full",
            "sd_model_full",
            "SD_model",
            "sd_model",
            "SD_model_oof_filtered_nva20",
            "sd_model_oof_filtered_nva20",
            "SD_model_oof_all",
            "sd_model_oof_all",
            "SD_oof",
            "sd_oof",
            "sigma_corrected",
            "sigma_model",
            "sd_corrected",
        ]

    def normalize_name(name):
        return str(name).strip().lower().replace(" ", "_")

    def extract_from_array(value):
        if value is None:
            return None

        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return None

        if arr.size == 0:
            return None

        if arr.size == 1:
            val = float(arr[0])
        elif arr.size > j:
            val = float(arr[j])
        else:
            return None

        if np.isfinite(val) and val > 0:
            return val

        return None

    def extract_from_dataframe(df):
        if df is None or len(df) == 0:
            return None

        df = df.copy()

        period_col = None
        for candidate in ["Period", "period", "T", "Periodo", "periods"]:
            if candidate in df.columns:
                period_col = candidate
                break

        if period_col is not None:
            df["_period_distance"] = np.abs(
                pd.to_numeric(df[period_col], errors="coerce") - selected_period
            )
            row = df.sort_values("_period_distance").head(1)
        else:
            row = df.iloc[[min(j, len(df) - 1)]]

        normalized_cols = {normalize_name(c): c for c in df.columns}

        for name in preferred_names:
            key = normalize_name(name)

            if key in normalized_cols:
                real_col = normalized_cols[key]
                val = pd.to_numeric(row[real_col], errors="coerce").iloc[0]

                if np.isfinite(val) and val > 0:
                    return float(val)

        return None

    # 1. Buscar primero en evaluation_curves["SD"], que es lo más probable.
    eval_curves = artifact.get("evaluation_curves", {})

    if isinstance(eval_curves, dict):
        for key in ["SD", "sd"]:
            if key in eval_curves:
                val = extract_from_dataframe(eval_curves[key])
                if val is not None:
                    return val

    # 2. Buscar en dataframes directos dentro del artefacto.
    for key in [
        "metrics_df",
        "comparison_curves",
        "curves",
        "sd_curves",
        "evaluation_df",
    ]:
        obj = artifact.get(key)

        if isinstance(obj, pd.DataFrame):
            val = extract_from_dataframe(obj)
            if val is not None:
                return val

        if isinstance(obj, dict):
            for sub_obj in obj.values():
                if isinstance(sub_obj, pd.DataFrame):
                    val = extract_from_dataframe(sub_obj)
                    if val is not None:
                        return val

    # 3. Buscar arrays directos dentro del artefacto.
    for name in preferred_names:
        if name in artifact:
            val = extract_from_array(artifact[name])
            if val is not None:
                return val

    # 4. Búsqueda flexible por nombre parecido.
    normalized_preferred = [normalize_name(x) for x in preferred_names]

    for key, value in artifact.items():
        key_norm = normalize_name(key)

        if any(pref in key_norm or key_norm in pref for pref in normalized_preferred):
            val = extract_from_array(value)
            if val is not None:
                return val

    return None


def add_sigma_band(
    fig,
    x,
    y,
    sigma,
    name,
    fillcolor,
    sigma_multiplier=1.0,
    legendgroup=None,
):
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return fig

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)

    if valid.sum() < 2:
        return fig

    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ksigma = float(sigma_multiplier) * float(sigma)

    upper = y * np.exp(ksigma)
    lower = y * np.exp(-ksigma)

    group = legendgroup or name

    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup=group,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fillcolor,
            name=f"{name} ±{float(sigma_multiplier):g}σ",
            hoverinfo="skip",
            legendgroup=group,
        )
    )

    return fig


def attenuation_figure(
    package,
    model_key,
    base_scenario,
    selected_period,
    apply_gate,
    gate_alpha,
    gate_beta,
    n_points,
    show_sigma_base=True,
    show_sigma_corrected=True,
    sigma_multiplier=1.0,
):
    result = predict_attenuation_curve(
        package=package,
        model_key=model_key,
        base_scenario=base_scenario,
        selected_period=selected_period,
        apply_gate=apply_gate,
        gate_alpha=gate_alpha,
        gate_beta=gate_beta,
        n_points=n_points,
    )

    artifact = package["models"][model_key]

    T = result["T"]
    rrup_grid = result["rrup_grid"]
    color = model_color(model_key)

    fig = go.Figure()

    obs = load_observed_for_attenuation(
        model_key=model_key,
        selected_period=T,
        mw=base_scenario["magnitude"],
        zhypo=base_scenario["zhypo_km"],
        soil=base_scenario["soil_class"],
        rvolc=base_scenario.get("rvolc_km", 0.0),
    )

    if not obs.empty:
        fig.add_trace(
            go.Scatter(
                x=obs["Rrup_km"],
                y=obs["Sa"],
                mode="markers",
                name="Observado en banda",
                marker=dict(
                    size=6,
                    color=NEUTRAL["observed"],
                    line=dict(color="white", width=0.4),
                ),
                opacity=0.70,
            )
        )

    sigma_base = find_sigma_for_period(artifact, T, curve_type="base")
    sigma_corrected = find_sigma_for_period(artifact, T, curve_type="corrected")

    if show_sigma_base:
        fig = add_sigma_band(
            fig,
            rrup_grid,
            result["base_sa_T"],
            sigma_base,
            f"{result['base_name']} base",
            fillcolor=NEUTRAL["sigma_base"],
            sigma_multiplier=float(sigma_multiplier),
            legendgroup="sigma_base",
        )

    if show_sigma_corrected:
        fig = add_sigma_band(
            fig,
            rrup_grid,
            result["corrected_sa_T"],
            sigma_corrected,
            f"{result['base_name']} + GRU corregido",
            fillcolor=model_soft_color(model_key),
            sigma_multiplier=float(sigma_multiplier),
            legendgroup="sigma_corrected",
        )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["base_sa_T"],
            mode="lines",
            name=f"{result['base_name']} base",
            line=dict(color=color, width=2.8, dash="dash"),
            legendgroup=model_key,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["corrected_sa_T"],
            mode="lines",
            name=f"{result['base_name']} + GRU corregido",
            line=dict(color=color, width=3.3),
            legendgroup=model_key,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["gate_weight"],
            mode="lines",
            name="peso gate",
            yaxis="y2",
            line=dict(color=NEUTRAL["gate"], width=2.6, dash="dot"),
        )
    )

    fig.update_xaxes(type="log", title="Rrup (km)")
    fig.update_yaxes(type="log", title=f"Sa(T={T:g} s) (g)")

    fig = apply_prediction_layout(
        fig,
        title=(
            f"Curva de atenuación, bandas ±{float(sigma_multiplier):g}σ "
            f"y peso del gate — T={T:g} s"
        ),
        height=740,
        legend_y=-0.28,
    )

    fig.update_layout(
        yaxis2=dict(
            title="w_gate",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            showgrid=False,
            linecolor=NEUTRAL["axis"],
            tickfont=dict(color="#374151"),
            title_font=dict(color=NEUTRAL["gate"]),
        )
    )

    if sigma_base is None and sigma_corrected is None:
        fig.add_annotation(
            text="No se encontraron curvas SD en model.pkl. La banda no fue graficada.",
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color=NEUTRAL["muted"]),
        )

    return fig


# =========================================================
# COMPONENTES DE FORMULARIO
# =========================================================

def scenario_form(prefix, title, subtitle=None):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="mb-1"),
                html.P(subtitle or "", className="text-muted small mb-3"),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Magnitud Mw", className="small"),
                                dbc.Input(
                                    id=f"{prefix}-magnitude",
                                    type="number",
                                    value=5.5,
                                    min=3.0,
                                    max=8.5,
                                    step=0.1,
                                    size="sm",
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Rrup (km)", className="small"),
                                dbc.Input(
                                    id=f"{prefix}-rrup",
                                    type="number",
                                    value=100.0,
                                    min=0.1,
                                    step=1.0,
                                    size="sm",
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Prof. hipocentral (km)", className="small"),
                                dbc.Input(
                                    id=f"{prefix}-zhypo",
                                    type="number",
                                    value=10.0,
                                    min=0.0,
                                    step=1.0,
                                    size="sm",
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Clase de suelo", className="small"),
                                dcc.Dropdown(
                                    id=f"{prefix}-soil",
                                    options=[{"label": str(i), "value": i} for i in range(1, 6)],
                                    value=2,
                                    clearable=False,
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Rvolc (km)", className="small"),
                                dbc.Input(
                                    id=f"{prefix}-rvolc",
                                    type="number",
                                    value=0.0,
                                    min=0.0,
                                    step=1.0,
                                    size="sm",
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Elevación estación (m)", className="small"),
                                dbc.Input(
                                    id=f"{prefix}-elev",
                                    type="number",
                                    value=100.0,
                                    min=0.0,
                                    step=10.0,
                                    size="sm",
                                ),
                            ],
                            xs=12,
                            sm=6,
                            lg=12,
                            xl=6,
                        ),
                    ],
                    className="g-2",
                ),
            ]
        ),
        className="soft-card mb-3",
    )


# =========================================================
# LAYOUT
# =========================================================

def layout():
    package = get_package()

    if isinstance(package, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("No hay modelo entrenado para predicción.", className="alert-heading"),
                    html.P(str(package)),
                    html.P(
                        "Ejecuta primero: "
                        "python model/train_model.py --models nosam,nosam_elevation,ask14 --epochs 60"
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

    default_alpha, default_beta = common_gate_default_params(default_artifact)

    return dbc.Container(
        [
            html.Div("Predicción interactiva", className="section-kicker"),
            html.H1("Predicción física del espectro corregido", className="page-title"),
            html.P(
                "Cada gráfica tiene su propio formulario y su propio botón. "
                "La configuración de modelo y gate se mantiene global. En las gráficas, "
                "la GMPE base y su versión corregida usan el mismo color; se diferencian "
                "por línea punteada vs línea continua.",
                className="lead",
            ),

            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Configuración global", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Modelo"),
                                        dcc.Dropdown(
                                            id="pred-model",
                                            options=opts,
                                            value=default_model,
                                            clearable=False,
                                        ),
                                    ],
                                    xs=12,
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Gate post hoc"),
                                        dbc.Checklist(
                                            id="pred-apply-gate",
                                            options=[{"label": "Aplicar gate", "value": "gate"}],
                                            value=["gate"],
                                            switch=True,
                                        ),
                                    ],
                                    xs=12,
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Alpha"),
                                        dbc.Input(
                                            id="pred-gate-alpha",
                                            type="number",
                                            value=default_alpha,
                                            step=0.1,
                                            min=0.01,
                                            placeholder="Auto",
                                        ),
                                    ],
                                    xs=12,
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Beta"),
                                        dbc.Input(
                                            id="pred-gate-beta",
                                            type="number",
                                            value=default_beta,
                                            step=0.1,
                                            min=0.01,
                                            placeholder="Auto",
                                        ),
                                    ],
                                    xs=12,
                                    md=3,
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
                        [
                            scenario_form(
                                "fixed",
                                "Escenario para espectro fijo",
                                "Estos valores solo actualizan la primera gráfica.",
                            ),
                            dbc.Button(
                                "Predecir espectro fijo",
                                id="fixed-button",
                                color="primary",
                                className="w-100 mt-2 mb-3",
                            ),
                        ],
                        xs=12,
                        lg=4,
                        xl=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(html.Div(id="prediction-summary")),
                                className="highlight-card mb-4",
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="prediction-single-graph",
                                        figure=empty_figure(
                                            "Presiona 'Predecir espectro fijo'.",
                                            height=640,
                                        ),
                                        style={"height": "660px"},
                                        config={"responsive": True},
                                    )
                                ),
                                className="soft-card",
                            ),
                        ],
                        xs=12,
                        lg=8,
                        xl=9,
                    ),
                ],
                className="g-4 mb-5 align-items-start",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            scenario_form(
                                "sens",
                                "Escenario base para sensibilidad",
                                "Estos valores solo actualizan la segunda gráfica.",
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Variable a cambiar", className="mb-3"),

                                        dcc.Dropdown(
                                            id="sens-var",
                                            options=variable_options(default_model),
                                            value="magnitude",
                                            clearable=False,
                                        ),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Mínimo", className="small mt-2"),
                                                        dbc.Input(
                                                            id="sens-var-min",
                                                            type="number",
                                                            value=4.5,
                                                            step=0.1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Máximo", className="small mt-2"),
                                                        dbc.Input(
                                                            id="sens-var-max",
                                                            type="number",
                                                            value=7.0,
                                                            step=0.1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("N", className="small mt-2"),
                                                        dbc.Input(
                                                            id="sens-var-n",
                                                            type="number",
                                                            value=4,
                                                            min=2,
                                                            max=8,
                                                            step=1,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=4,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                    ]
                                ),
                                className="soft-card mb-3",
                            ),
                            dbc.Button(
                                "Actualizar sensibilidad",
                                id="sens-button",
                                color="secondary",
                                className="w-100 mt-2 mb-3",
                            ),
                        ],
                        xs=12,
                        lg=4,
                        xl=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(
                                    id="prediction-sensitivity-graph",
                                    figure=empty_figure(
                                        "Presiona 'Actualizar sensibilidad'.",
                                        height=700,
                                    ),
                                    style={"height": "720px"},
                                    config={"responsive": True},
                                )
                            ),
                            className="soft-card",
                        ),
                        xs=12,
                        lg=8,
                        xl=9,
                    ),
                ],
                className="g-4 mb-5 align-items-start",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            scenario_form(
                                "atten",
                                "Escenario para atenuación",
                                "Estos valores solo actualizan la curva de atenuación.",
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Configuración de atenuación", className="mb-3"),

                                        html.Label("Período", className="small"),
                                        dcc.Dropdown(
                                            id="atten-period",
                                            options=p_opts,
                                            value=default_period,
                                            clearable=False,
                                        ),

                                        html.Label("Puntos de curva", className="small mt-3"),
                                        dbc.Input(
                                            id="atten-n-points",
                                            type="number",
                                            value=35,
                                            min=10,
                                            max=120,
                                            step=5,
                                            size="sm",
                                        ),

                                        html.Label("Bandas de desviación estándar", className="small mt-3"),
                                        dbc.Checklist(
                                            id="atten-sigma-bands",
                                            options=[
                                                {"label": "Banda GMPE base", "value": "base"},
                                                {"label": "Banda corregida", "value": "corrected"},
                                            ],
                                            value=["base", "corrected"],
                                            switch=True,
                                        ),

                                        html.Label("Multiplicador sigma", className="small mt-3"),
                                        dcc.Dropdown(
                                            id="atten-sigma-multiplier",
                                            options=[
                                                {"label": "±1σ", "value": 1.0},
                                                {"label": "±2σ", "value": 2.0},
                                            ],
                                            value=1.0,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                className="soft-card mb-3",
                            ),
                            dbc.Button(
                                "Actualizar curva de atenuación",
                                id="atten-button",
                                color="dark",
                                className="w-100 mt-2 mb-3",
                            ),
                        ],
                        xs=12,
                        lg=4,
                        xl=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(
                                    id="prediction-attenuation-graph",
                                    figure=empty_figure(
                                        "Presiona 'Actualizar curva de atenuación'.",
                                        height=740,
                                    ),
                                    style={"height": "760px"},
                                    config={"responsive": True},
                                )
                            ),
                            className="soft-card",
                        ),
                        xs=12,
                        lg=8,
                        xl=9,
                    ),
                ],
                className="g-4 align-items-start",
            ),
        ],
        fluid=True,
    )


# =========================================================
# CALLBACKS
# =========================================================

def register_callbacks(app):
    @app.callback(
        Output("sens-var", "options"),
        Output("sens-var", "value"),
        Output("atten-period", "options"),
        Output("atten-period", "value"),
        Output("pred-gate-alpha", "value"),
        Output("pred-gate-beta", "value"),
        Input("pred-model", "value"),
    )
    def update_global_model_options(model_key):
        package = get_package()

        if isinstance(package, Exception):
            return [], None, [], None, None, None

        if model_key not in package.get("models", {}):
            return [], None, [], None, None, None

        artifact = package["models"][model_key]

        p_opts = period_options(artifact)
        v_opts = variable_options(model_key)

        alpha, beta = common_gate_default_params(artifact)

        return (
            v_opts,
            v_opts[0]["value"] if v_opts else None,
            p_opts,
            p_opts[0]["value"] if p_opts else None,
            alpha,
            beta,
        )

    @app.callback(
        Output("prediction-summary", "children"),
        Output("prediction-single-graph", "figure"),
        Input("fixed-button", "n_clicks"),
        State("pred-model", "value"),
        State("pred-apply-gate", "value"),
        State("pred-gate-alpha", "value"),
        State("pred-gate-beta", "value"),
        State("fixed-magnitude", "value"),
        State("fixed-rrup", "value"),
        State("fixed-zhypo", "value"),
        State("fixed-soil", "value"),
        State("fixed-rvolc", "value"),
        State("fixed-elev", "value"),
        prevent_initial_call=True,
    )
    def update_fixed_spectrum(
        n_clicks,
        model_key,
        gate_values,
        gate_alpha,
        gate_beta,
        magnitude,
        rrup,
        zhypo,
        soil,
        rvolc,
        elev,
    ):
        if not n_clicks:
            raise PreventUpdate

        package = get_package()

        if isinstance(package, Exception):
            fig = empty_figure("Modelo no disponible.")
            return dbc.Alert(str(package), color="warning"), fig

        try:
            apply_gate = "gate" in (gate_values or [])

            sc = scenario_dict(
                magnitude=magnitude,
                rrup=rrup,
                zhypo=zhypo,
                soil=soil,
                rvolc=rvolc,
                elev=elev,
            )

            result = predict_many(
                package=package,
                model_key=model_key,
                scenarios=[sc],
                apply_gate=apply_gate,
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
            )

            artifact = package["models"][model_key]
            gate_text = gate_info_text(result["gate_obj"], apply_gate)

            summary = [
                html.H4(artifact.get("label", model_key), className="mb-2"),
                html.P(f"GMPE base: {result['base_name']}"),
                html.P(gate_text),
                html.P(f"w_gate escenario fijo = {result['gate_weight'][0]:.3f}"),
                html.P(
                    f"Primer período: T={result['periods'][0]:.3g} s | "
                    f"Sa base={result['base_sa'][0, 0]:.4g} | "
                    f"Sa corregido={result['corrected_sa'][0, 0]:.4g}",
                    className="mb-0",
                ),
            ]

            return summary, spectrum_single_figure(result, model_key)

        except Exception as exc:
            fig = empty_figure(str(exc))
            return dbc.Alert(str(exc), color="danger"), fig

    @app.callback(
        Output("prediction-sensitivity-graph", "figure"),
        Input("sens-button", "n_clicks"),
        State("pred-model", "value"),
        State("pred-apply-gate", "value"),
        State("pred-gate-alpha", "value"),
        State("pred-gate-beta", "value"),
        State("sens-magnitude", "value"),
        State("sens-rrup", "value"),
        State("sens-zhypo", "value"),
        State("sens-soil", "value"),
        State("sens-rvolc", "value"),
        State("sens-elev", "value"),
        State("sens-var", "value"),
        State("sens-var-min", "value"),
        State("sens-var-max", "value"),
        State("sens-var-n", "value"),
        prevent_initial_call=True,
    )
    def update_sensitivity_graph(
        n_clicks,
        model_key,
        gate_values,
        gate_alpha,
        gate_beta,
        magnitude,
        rrup,
        zhypo,
        soil,
        rvolc,
        elev,
        var_name,
        vmin,
        vmax,
        nvals,
    ):
        if not n_clicks:
            raise PreventUpdate

        package = get_package()

        if isinstance(package, Exception):
            return empty_figure(str(package))

        try:
            apply_gate = "gate" in (gate_values or [])

            base_sc = scenario_dict(
                magnitude=magnitude,
                rrup=rrup,
                zhypo=zhypo,
                soil=soil,
                rvolc=rvolc,
                elev=elev,
            )

            values = sensitivity_values(
                var_name=var_name,
                vmin=vmin,
                vmax=vmax,
                n=nvals,
            )

            scenarios = []

            for val in values:
                sc = base_sc.copy()

                if var_name == "soil_class":
                    sc[var_name] = int(val)
                else:
                    sc[var_name] = float(val)

                scenarios.append(sc)

            result = predict_many(
                package=package,
                model_key=model_key,
                scenarios=scenarios,
                apply_gate=apply_gate,
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
            )

            return sensitivity_figure(result, var_name, values, model_key)

        except Exception as exc:
            return empty_figure(str(exc))

    @app.callback(
        Output("prediction-attenuation-graph", "figure"),
        Input("atten-button", "n_clicks"),
        State("pred-model", "value"),
        State("pred-apply-gate", "value"),
        State("pred-gate-alpha", "value"),
        State("pred-gate-beta", "value"),
        State("atten-magnitude", "value"),
        State("atten-rrup", "value"),
        State("atten-zhypo", "value"),
        State("atten-soil", "value"),
        State("atten-rvolc", "value"),
        State("atten-elev", "value"),
        State("atten-period", "value"),
        State("atten-n-points", "value"),
        State("atten-sigma-bands", "value"),
        State("atten-sigma-multiplier", "value"),
        prevent_initial_call=True,
    )
    def update_attenuation_graph(
        n_clicks,
        model_key,
        gate_values,
        gate_alpha,
        gate_beta,
        magnitude,
        rrup,
        zhypo,
        soil,
        rvolc,
        elev,
        atten_period,
        n_points,
        sigma_bands,
        sigma_multiplier,
    ):
        if not n_clicks:
            raise PreventUpdate

        package = get_package()

        if isinstance(package, Exception):
            return empty_figure(str(package))

        try:
            apply_gate = "gate" in (gate_values or [])

            base_sc = scenario_dict(
                magnitude=magnitude,
                rrup=rrup,
                zhypo=zhypo,
                soil=soil,
                rvolc=rvolc,
                elev=elev,
            )

            if atten_period is None:
                artifact = package["models"][model_key]
                periods = np.asarray(artifact["periods"], dtype=float)
                atten_period = float(periods[0])

            sigma_bands = sigma_bands or []

            return attenuation_figure(
                package=package,
                model_key=model_key,
                base_scenario=base_sc,
                selected_period=float(atten_period),
                apply_gate=apply_gate,
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
                n_points=int(n_points or 35),
                show_sigma_base="base" in sigma_bands,
                show_sigma_corrected="corrected" in sigma_bands,
                sigma_multiplier=float(sigma_multiplier or 1.0),
            )

        except Exception as exc:
            return empty_figure(str(exc))