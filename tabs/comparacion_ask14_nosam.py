from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from tabs.decomp_variabilidad import (
    decomposition_card,
    register_decomposition_callbacks,
    gate_alpha_dropdown_options,
    gate_beta_dropdown_options,
    default_cached_gate_params,
)

MODEL_KEY_RESIDUAL = "ask14"
COMMON_GATE_COLS = ["Rrup_km", "Magnitude", "Hypocenter Depth (km)"]
DEFAULT_ALPHA = 3.0
DEFAULT_BETA = 3.0

VARIABLE_OPTIONS = [
    {"label": "Magnitud Mw", "value": "magnitude"},
    {"label": "Rrup (km)", "value": "rrup_km"},
    {"label": "Profundidad hipocentral (km)", "value": "zhypo_km"},
    {"label": "Clase de suelo", "value": "soil_class"},
    {"label": "Rvolc (km)", "value": "rvolc_km"},
]

VARIABLE_LABELS = {
    "magnitude": "Magnitud Mw",
    "rrup_km": "Rrup (km)",
    "zhypo_km": "Profundidad hipocentral (km)",
    "soil_class": "Clase de suelo",
    "rvolc_km": "Rvolc (km)",
}

COLORS = {
    "ask14": "#1D4ED8",
    "ask14_soft": "rgba(29, 78, 216, 0.20)",
    "nosam": "#1F2937",
    "nosam_soft": "rgba(31, 41, 55, 0.16)",
    "gate": "#B45309",
    "text": "#111827",
    "muted": "#64748B",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
}

SCENARIO_COLORS = [
    "#1F2937",
    "#334155",
    "#0F766E",
    "#1D4ED8",
    "#7C3AED",
    "#B45309",
    "#9F1239",
    "#475569",
    "#155E75",
    "#78350F",
]

_PACKAGE_CACHE = None
_KERAS_MODEL_CACHE = {}


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


def scenario_dict(magnitude, rrup, zhypo, soil, rvolc):
    return {
        "magnitude": parse_number(magnitude, 5.5),
        "rrup_km": parse_number(rrup, 100.0),
        "zhypo_km": parse_number(zhypo, 10.0),
        "soil_class": int(parse_number(soil, 2)),
        "rvolc_km": parse_number(rvolc, 0.0),
    }


def varied_values(var_name, vmin, vmax, n):
    n = max(2, min(int(parse_number(n, 4)), 10))
    vmin = parse_number(vmin, 0.0)
    vmax = parse_number(vmax, 1.0)

    if vmax <= vmin:
        vmax = vmin + 1.0

    if var_name == "soil_class":
        vals = [min(max(int(round(v)), 1), 5) for v in np.linspace(vmin, vmax, n)]
        return sorted(list(dict.fromkeys(vals)))

    if var_name == "rrup_km":
        return np.geomspace(max(vmin, 0.1), max(vmax, 0.2), n)

    return np.linspace(vmin, vmax, n)


def period_options(artifact):
    periods = np.asarray(artifact.get("periods", []), dtype=float)
    return [{"label": f"T = {p:g} s", "value": float(p)} for p in periods]


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


def apply_compare_layout(fig, title, height=760, legend_y=-0.28):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=45, r=35, t=70, b=105),
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            size=13,
            color=COLORS["text"],
        ),
        title_font=dict(size=18, color=COLORS["text"]),
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
        gridcolor=COLORS["grid"],
        zeroline=False,
        linecolor=COLORS["axis"],
        autorange=True,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        linecolor=COLORS["axis"],
        autorange=True,
    )

    return fig


def empty_figure(message, height=760):
    fig = go.Figure()

    fig.add_annotation(
        text=str(message),
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=COLORS["muted"]),
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_visible=False,
        yaxis_visible=False,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def _subset_standard_scaler(scaler, old_cols, new_cols):
    new_scaler = deepcopy(scaler)
    idx = [list(old_cols).index(c) for c in new_cols]
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
    if gate_obj is None:
        return None

    gate = deepcopy(gate_obj)

    old_cols = list(gate.get("gate_cont_cols", COMMON_GATE_COLS))
    new_cols = COMMON_GATE_COLS.copy()

    missing = [c for c in new_cols if c not in old_cols]

    if missing:
        raise ValueError(
            f"El gate no contiene las variables comunes requeridas. "
            f"Faltan: {missing}. Columnas actuales: {old_cols}"
        )

    if old_cols != new_cols:
        gate["scaler"] = _subset_standard_scaler(gate["scaler"], old_cols, new_cols)

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
            "El artefacto ASK14 no contiene objeto gate. "
            "Reentrena el modelo o desactiva el gate."
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
        return "Gate aplicado: no | corrección residual ASK14 completa"

    if gate_obj is None:
        return "Gate aplicado: no disponible"

    original = gate_obj.get("original_gate_cont_cols")
    original_txt = ""

    if original and original != gate_obj.get("gate_cont_cols"):
        original_txt = f" | gate original={original}"

    return (
        f"Gate aplicado: sí | "
        f"variables usadas={gate_obj.get('gate_cont_cols')} | "
        f"alpha={fmt_number(gate_obj.get('alpha'))} | "
        f"beta={fmt_number(gate_obj.get('beta'))}"
        f"{original_txt}"
    )


def predict_compare_spectra(
    package,
    scenarios,
    apply_gate=True,
    gate_alpha=None,
    gate_beta=None,
):
    from model.train_model import (
        build_prediction_frame,
        build_sequence_input,
        base_sa_spectrum,
        gate_weight_single,
    )

    if MODEL_KEY_RESIDUAL not in package.get("models", {}):
        raise ValueError(
            "model.pkl no contiene el modelo 'ask14'. "
            "Entrena primero: python model/train_model.py --models ask14"
        )

    artifact = package["models"][MODEL_KEY_RESIDUAL]

    missing = [
        k for k in ["periods", "model", "preprocessor", "feature_cols"]
        if k not in artifact
    ]

    if missing:
        raise ValueError(
            f"El artefacto ASK14 no contiene estas claves: {missing}. "
            "Reentrena con train_model.py actualizado."
        )

    periods = np.asarray(artifact["periods"], dtype=float)
    model = get_cached_keras_model(MODEL_KEY_RESIDUAL, artifact)
    preprocessor = artifact["preprocessor"]

    gate_obj = make_gate_object(
        artifact,
        apply_gate,
        gate_alpha,
        gate_beta,
    )

    x_frames = []

    for sc in scenarios:
        frame = build_prediction_frame(
            model_key="ask14",
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
            station_elevation_m=100.0,
        )
        x_frames.append(frame)

    X_df = pd.concat(x_frames, ignore_index=True)

    missing_features = [
        c for c in artifact["feature_cols"]
        if c not in X_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Faltan columnas para predecir ASK14 residual: {missing_features}. "
            f"Columnas disponibles: {list(X_df.columns)}"
        )

    Xp = preprocessor.transform(X_df[artifact["feature_cols"]])

    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    X_seq = build_sequence_input(
        np.asarray(Xp, dtype=np.float32),
        periods,
    )

    residual_hat = np.asarray(model.predict(X_seq, verbose=0), dtype=float)

    if residual_hat.ndim == 3 and residual_hat.shape[-1] == 1:
        residual_hat = residual_hat[:, :, 0]

    if residual_hat.ndim == 1:
        residual_hat = residual_hat.reshape(1, -1)

    ask14_base_list = []
    ask14_corr_list = []
    nosam_list = []
    gate_list = []

    for i, sc in enumerate(scenarios):
        ask14_base, _ = base_sa_spectrum(
            model_key="ask14",
            periods=periods,
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        nosam_sa, _ = base_sa_spectrum(
            model_key="nosam",
            periods=periods,
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        if apply_gate:
            w_gate = gate_weight_single(
                gate_obj=gate_obj,
                model_key="ask14",
                periods=periods,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
            )
        else:
            w_gate = 1.0

        ask14_base = np.asarray(ask14_base, dtype=float)
        nosam_sa = np.asarray(nosam_sa, dtype=float)
        ask14_corr = ask14_base * np.exp(float(w_gate) * residual_hat[i, :])

        ask14_base_list.append(ask14_base)
        ask14_corr_list.append(ask14_corr)
        nosam_list.append(nosam_sa)
        gate_list.append(float(w_gate))

    return {
        "artifact": artifact,
        "periods": periods,
        "ask14_base": np.asarray(ask14_base_list, dtype=float),
        "ask14_corrected": np.asarray(ask14_corr_list, dtype=float),
        "nosam": np.asarray(nosam_list, dtype=float),
        "gate_weight": np.asarray(gate_list, dtype=float),
        "gate_obj": gate_obj,
        "apply_gate": bool(apply_gate),
        "prediction_frame": X_df,
    }


def predict_compare_attenuation(
    package,
    base_scenario,
    selected_period,
    apply_gate=True,
    gate_alpha=None,
    gate_beta=None,
    n_points=35,
    rrup_min=5.0,
    rrup_max=400.0,
):
    from model.train_model import (
        build_prediction_frame,
        build_sequence_input,
        base_sa_spectrum,
        gate_weight_single,
    )

    artifact = package["models"][MODEL_KEY_RESIDUAL]
    periods = np.asarray(artifact["periods"], dtype=float)

    j = int(np.argmin(np.abs(periods - float(selected_period))))
    T = float(periods[j])

    model = get_cached_keras_model(MODEL_KEY_RESIDUAL, artifact)
    preprocessor = artifact["preprocessor"]

    gate_obj = make_gate_object(
        artifact,
        apply_gate,
        gate_alpha,
        gate_beta,
    )

    n_points = max(10, min(int(parse_number(n_points, 35)), 120))

    rrup_min = max(parse_number(rrup_min, 5.0), 0.1)
    rrup_max = parse_number(rrup_max, 400.0)

    if rrup_max <= rrup_min:
        rrup_max = rrup_min * 1.5

    rrup_grid = np.geomspace(rrup_min, rrup_max, n_points)

    scenarios = []

    for rr in rrup_grid:
        sc = base_scenario.copy()
        sc["rrup_km"] = float(rr)
        scenarios.append(sc)

    x_frames = []

    for sc in scenarios:
        frame = build_prediction_frame(
            model_key="ask14",
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
            station_elevation_m=100.0,
        )
        x_frames.append(frame)

    X_df = pd.concat(x_frames, ignore_index=True)

    Xp = preprocessor.transform(X_df[artifact["feature_cols"]])

    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    X_seq = build_sequence_input(
        np.asarray(Xp, dtype=np.float32),
        periods,
    )

    residual_hat = np.asarray(model.predict(X_seq, verbose=0), dtype=float)

    if residual_hat.ndim == 3 and residual_hat.shape[-1] == 1:
        residual_hat = residual_hat[:, :, 0]

    if residual_hat.ndim == 1:
        residual_hat = residual_hat.reshape(1, -1)

    ask14_base = []
    ask14_corr = []
    nosam = []
    gate_weights = []

    for i, sc in enumerate(scenarios):
        ask14_base_T, _ = base_sa_spectrum(
            model_key="ask14",
            periods=np.array([T], dtype=float),
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        nosam_T, _ = base_sa_spectrum(
            model_key="nosam",
            periods=np.array([T], dtype=float),
            magnitude=sc["magnitude"],
            rrup_km=sc["rrup_km"],
            zhypo_km=sc["zhypo_km"],
            soil_class=sc["soil_class"],
            rvolc_km=sc.get("rvolc_km", 0.0),
        )

        ask14_base_val = float(np.asarray(ask14_base_T).reshape(-1)[0])
        nosam_val = float(np.asarray(nosam_T).reshape(-1)[0])

        if apply_gate:
            w_gate = gate_weight_single(
                gate_obj=gate_obj,
                model_key="ask14",
                periods=periods,
                magnitude=sc["magnitude"],
                rrup_km=sc["rrup_km"],
                zhypo_km=sc["zhypo_km"],
                soil_class=sc["soil_class"],
                rvolc_km=sc.get("rvolc_km", 0.0),
            )
        else:
            w_gate = 1.0

        ask14_corr_val = ask14_base_val * np.exp(
            float(w_gate) * float(residual_hat[i, j])
        )

        ask14_base.append(ask14_base_val)
        ask14_corr.append(ask14_corr_val)
        nosam.append(nosam_val)
        gate_weights.append(float(w_gate))

    return {
        "T": T,
        "rrup_grid": rrup_grid,
        "ask14_base": np.asarray(ask14_base, dtype=float),
        "ask14_corrected": np.asarray(ask14_corr, dtype=float),
        "nosam": np.asarray(nosam, dtype=float),
        "gate_weight": np.asarray(gate_weights, dtype=float),
        "gate_obj": gate_obj,
    }


def find_sigma_for_period(artifact, selected_period, curve_type="corrected"):
    periods = np.asarray(artifact.get("periods", []), dtype=float)

    if periods.size == 0:
        return None

    j = int(np.argmin(np.abs(periods - float(selected_period))))

    if curve_type == "base":
        names = [
            "SD_base",
            "sd_base",
            "sigma_base",
            "gmpe_sigma",
            "sd_gmpe",
        ]
    else:
        names = [
            "SD_model_full_train",
            "sd_model_full_train",
            "SD_model_full",
            "sd_model_full",
            "SD_model",
            "sd_model",
            "SD_oof",
            "sd_oof",
            "sigma_corrected",
            "sigma_model",
            "sd_corrected",
        ]

    def from_array(value):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)

            if arr.size == 1:
                val = float(arr[0])
            elif arr.size > j:
                val = float(arr[j])
            else:
                return None

            return val if np.isfinite(val) and val > 0 else None

        except Exception:
            return None

    def from_df(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        df = df.copy()

        pcol = next(
            (
                c for c in ["Period", "period", "T", "Periodo", "periods"]
                if c in df.columns
            ),
            None,
        )

        if pcol is None:
            row = df.iloc[[min(j, len(df) - 1)]]
        else:
            dist = np.abs(
                pd.to_numeric(df[pcol], errors="coerce") - float(selected_period)
            )
            row = df.iloc[[int(np.nanargmin(dist))]]

        cols = {
            str(c).strip().lower().replace(" ", "_"): c
            for c in df.columns
        }

        for name in names:
            key = name.lower().replace(" ", "_")

            if key in cols:
                val = pd.to_numeric(row[cols[key]], errors="coerce").iloc[0]

                if np.isfinite(val) and val > 0:
                    return float(val)

        return None

    eval_curves = artifact.get("evaluation_curves", {})

    if isinstance(eval_curves, dict):
        for obj in eval_curves.values():
            val = from_df(obj)

            if val is not None:
                return val

    for key in [
        "metrics_df",
        "comparison_curves",
        "curves",
        "sd_curves",
        "evaluation_df",
    ]:
        obj = artifact.get(key)

        if isinstance(obj, pd.DataFrame):
            val = from_df(obj)

            if val is not None:
                return val

        if isinstance(obj, dict):
            for sub in obj.values():
                val = from_df(sub)

                if val is not None:
                    return val

    for name in names:
        if name in artifact:
            val = from_array(artifact[name])

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

    upper = y * np.exp(float(sigma_multiplier) * float(sigma))
    lower = y * np.exp(-float(sigma_multiplier) * float(sigma))

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


def multi_spectra_figure(result, variable_name, values, base_scenario=None):
    periods = result["periods"]
    fig = go.Figure()

    variable_label = VARIABLE_LABELS.get(variable_name, variable_name)

    for i, value in enumerate(values):
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]

        if variable_name == "soil_class":
            label = f"{variable_label}={int(value)}"
        else:
            label = f"{variable_label}={float(value):g}"

        group = f"scenario-{i}"

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=result["ask14_corrected"][i],
                mode="lines+markers",
                name=f"ASK14 residual corregido | {label} | w={result['gate_weight'][i]:.2f}",
                line=dict(color=color, width=3.1),
                marker=dict(size=6, color=color),
                legendgroup=group,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=result["nosam"][i],
                mode="lines",
                name=f"NoSAm puro | {label}",
                line=dict(color=color, width=2.4, dash="dash"),
                legendgroup=group,
            )
        )

    fig.update_xaxes(
        type="log",
        title="Período T (s)",
        autorange=True,
    )

    fig.update_yaxes(
        type="log",
        title="Sa (g)",
        autorange=True,
    )

    fig = apply_compare_layout(
        fig,
        "Múltiples espectros: ASK14 residual corregido vs NoSAm puro",
        height=760,
        legend_y=-0.34,
    )

    fig.update_xaxes(
        type="log",
        title="Período T (s)",
        autorange=True,
    )

    fig.update_yaxes(
        type="log",
        title="Sa (g)",
        autorange=True,
    )

    if base_scenario is not None:
        fixed_txt = (
            f"Mw={base_scenario['magnitude']:g}, "
            f"Rrup={base_scenario['rrup_km']:g} km, "
            f"Zhypo={base_scenario['zhypo_km']:g} km, "
            f"Soil={base_scenario['soil_class']}, "
            f"Rvolc={base_scenario['rvolc_km']:g} km"
        )

        fig.add_annotation(
            text=f"Escenario base: {fixed_txt}",
            x=0.01,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=12, color=COLORS["muted"]),
        )

    return fig


def attenuation_compare_figure(
    package,
    result,
    show_sigma_ask14=True,
    show_sigma_nosam=False,
    sigma_multiplier=1.0,
):
    T = result["T"]
    rrup_grid = result["rrup_grid"]

    fig = go.Figure()

    ask14_artifact = package.get("models", {}).get("ask14", {})
    nosam_artifact = package.get("models", {}).get("nosam", {})

    sigma_ask14 = find_sigma_for_period(
        ask14_artifact,
        T,
        "corrected",
    )

    sigma_nosam = (
        find_sigma_for_period(nosam_artifact, T, "base")
        if nosam_artifact
        else None
    )

    if show_sigma_nosam:
        fig = add_sigma_band(
            fig,
            rrup_grid,
            result["nosam"],
            sigma_nosam,
            "NoSAm puro",
            COLORS["nosam_soft"],
            sigma_multiplier,
            "sigma-nosam",
        )

    if show_sigma_ask14:
        fig = add_sigma_band(
            fig,
            rrup_grid,
            result["ask14_corrected"],
            sigma_ask14,
            "ASK14 residual corregido",
            COLORS["ask14_soft"],
            sigma_multiplier,
            "sigma-ask14",
        )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["ask14_corrected"],
            mode="lines",
            name="ASK14 residual corregido",
            line=dict(color=COLORS["ask14"], width=3.3),
            legendgroup="ask14",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["nosam"],
            mode="lines",
            name="NoSAm puro",
            line=dict(color=COLORS["nosam"], width=2.9, dash="dash"),
            legendgroup="nosam",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rrup_grid,
            y=result["gate_weight"],
            mode="lines",
            name="peso gate ASK14 residual",
            yaxis="y2",
            line=dict(color=COLORS["gate"], width=2.6, dash="dot"),
        )
    )

    fig.update_xaxes(
        type="log",
        title="Rrup (km)",
        autorange=True,
    )

    fig.update_yaxes(
        type="log",
        title=f"Sa(T={T:g} s) (g)",
        autorange=True,
    )

    fig = apply_compare_layout(
        fig,
        f"Curva de atenuación: ASK14 residual corregido vs NoSAm puro — T={T:g} s",
        height=760,
        legend_y=-0.28,
    )

    fig.update_xaxes(
        type="log",
        title="Rrup (km)",
        autorange=True,
    )

    fig.update_yaxes(
        type="log",
        title=f"Sa(T={T:g} s) (g)",
        autorange=True,
    )

    fig.update_layout(
        yaxis2=dict(
            title="w_gate",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            showgrid=False,
            linecolor=COLORS["axis"],
            title_font=dict(color=COLORS["gate"]),
        )
    )

    missing = []

    if show_sigma_ask14 and sigma_ask14 is None:
        missing.append("No se encontró SD corregida de ASK14 para sombrear banda.")

    if show_sigma_nosam and sigma_nosam is None:
        missing.append("No se encontró SD base de NoSAm para sombrear banda.")

    if missing:
        fig.add_annotation(
            text="<br>".join(missing),
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )

    return fig


def spec_scenario_form():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Escenario base para múltiples espectros", className="mb-1"),
                html.P(
                    "Define el escenario base y luego selecciona una variable para cambiar.",
                    className="text-muted small mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Magnitud Mw", className="small"),
                                dbc.Input(
                                    id="cmp-spec-fixed-magnitude",
                                    type="text",
                                    value="5.5",
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
                                    id="cmp-spec-fixed-rrup",
                                    type="text",
                                    value="100",
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
                                    id="cmp-spec-fixed-zhypo",
                                    type="text",
                                    value="10",
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
                                    id="cmp-spec-fixed-soil",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 6)
                                    ],
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
                                    id="cmp-spec-fixed-rvolc",
                                    type="text",
                                    value="0",
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
                html.Small(
                    "Nota: estos campos son de texto para aceptar coma decimal y evitar que Dash envíe valores None.",
                    className="text-muted",
                ),
            ]
        ),
        className="soft-card mb-3",
    )


def attenuation_scenario_form():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Escenario para curva de atenuación", className="mb-1"),
                html.P(
                    "La magnitud, profundidad, suelo y Rvolc quedan fijos; Rrup se controla con el rango de la grilla.",
                    className="text-muted small mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Magnitud Mw", className="small"),
                                dbc.Input(
                                    id="cmp-atten-magnitude",
                                    type="text",
                                    value="5.5",
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
                                    id="cmp-atten-zhypo",
                                    type="text",
                                    value="10",
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
                                    id="cmp-atten-soil",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 6)
                                    ],
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
                                    id="cmp-atten-rvolc",
                                    type="text",
                                    value="0",
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


def layout():
    package = get_package()

    if isinstance(package, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("No se pudo cargar model.pkl.", className="alert-heading"),
                    html.P(str(package)),
                ],
                color="warning",
            ),
            fluid=True,
        )

    if MODEL_KEY_RESIDUAL not in package.get("models", {}):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5(
                        "No existe el modelo ASK14 residual en model.pkl.",
                        className="alert-heading",
                    ),
                    html.P("Entrena primero el modelo ASK14:"),
                    html.Code("python model/train_model.py --models ask14"),
                ],
                color="danger",
            ),
            fluid=True,
        )

    artifact = package["models"][MODEL_KEY_RESIDUAL]

    p_opts = period_options(artifact)
    default_period = p_opts[0]["value"] if p_opts else None

    default_alpha, default_beta = common_gate_default_params(artifact)

    default_alpha, default_beta = default_cached_gate_params(
        alpha_default=default_alpha or DEFAULT_ALPHA,
        beta_default=default_beta or DEFAULT_BETA,
    )

    alpha_options = gate_alpha_dropdown_options(beta=default_beta)
    beta_options = gate_beta_dropdown_options()

    return dbc.Container(
        [
            html.Div("Comparación de modelos", className="section-kicker"),
            html.H1("ASK14 residual corregido vs NoSAm puro", className="page-title"),
            html.P(
                "Esta página compara exclusivamente el espectro corregido del modelo residual ASK14 "
                "contra la predicción directa de NoSAm. El gate solo afecta la corrección residual de ASK14 "
                "y se fuerza a usar solo Magnitude, Rrup y profundidad hipocentral.",
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
                                        html.Label("Modelo residual"),
                                        dbc.Input(
                                            value="ASK14 + GRU residual",
                                            disabled=True,
                                        ),
                                    ],
                                    xs=12,
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Comparador"),
                                        dbc.Input(
                                            value="NoSAm puro",
                                            disabled=True,
                                        ),
                                    ],
                                    xs=12,
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Gate post hoc"),
                                        dbc.Checklist(
                                            id="cmp-apply-gate",
                                            options=[
                                                {
                                                    "label": "Aplicar gate",
                                                    "value": "gate",
                                                }
                                            ],
                                            value=["gate"],
                                            switch=True,
                                        ),
                                    ],
                                    xs=12,
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Alpha"),
                                        dcc.Dropdown(
                                            id="cmp-gate-alpha",
                                            options=alpha_options,
                                            value=default_alpha,
                                            clearable=False,
                                        ),
                                        html.Small(
                                            "Valores precalculados en decomp_cache.pkl",
                                            className="text-muted",
                                        ),
                                    ],
                                    xs=12,
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Beta"),
                                        dcc.Dropdown(
                                            id="cmp-gate-beta",
                                            options=beta_options,
                                            value=default_beta,
                                            clearable=False,
                                            disabled=len(beta_options) <= 1,
                                        ),
                                        html.Small(
                                            "Beta del gate precalculado; normalmente corresponde al percentil 95.",
                                            className="text-muted",
                                        ),
                                    ],
                                    xs=12,
                                    md=6,
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
                            spec_scenario_form(),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Variable a variar", className="mb-3"),
                                        dcc.Dropdown(
                                            id="cmp-spec-var",
                                            options=VARIABLE_OPTIONS,
                                            value="magnitude",
                                            clearable=False,
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Mín",
                                                            className="small mt-2",
                                                        ),
                                                        dbc.Input(
                                                            id="cmp-spec-var-min",
                                                            type="text",
                                                            value="4.5",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Máx",
                                                            className="small mt-2",
                                                        ),
                                                        dbc.Input(
                                                            id="cmp-spec-var-max",
                                                            type="text",
                                                            value="7",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "N",
                                                            className="small mt-2",
                                                        ),
                                                        dbc.Input(
                                                            id="cmp-spec-var-n",
                                                            type="text",
                                                            value="4",
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
                                "Comparar múltiples espectros",
                                id="cmp-spec-button",
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
                                dbc.CardBody(html.Div(id="cmp-spec-summary")),
                                className="highlight-card mb-4",
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="cmp-spec-graph",
                                        figure=empty_figure(
                                            "Presiona 'Comparar múltiples espectros'.",
                                            height=760,
                                        ),
                                        style={"height": "780px"},
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
                            attenuation_scenario_form(),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Configuración de atenuación",
                                            className="mb-3",
                                        ),
                                        html.Label("Período", className="small"),
                                        dcc.Dropdown(
                                            id="cmp-atten-period",
                                            options=p_opts,
                                            value=default_period,
                                            clearable=False,
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Rrup mínimo (km)",
                                                            className="small mt-3",
                                                        ),
                                                        dbc.Input(
                                                            id="cmp-atten-rrup-min",
                                                            type="text",
                                                            value="5",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Rrup máximo (km)",
                                                            className="small mt-3",
                                                        ),
                                                        dbc.Input(
                                                            id="cmp-atten-rrup-max",
                                                            type="text",
                                                            value="400",
                                                            size="sm",
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=6,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                        html.Label(
                                            "Puntos de curva",
                                            className="small mt-3",
                                        ),
                                        dbc.Input(
                                            id="cmp-atten-n-points",
                                            type="text",
                                            value="35",
                                            size="sm",
                                        ),
                                        html.Label(
                                            "Bandas de desviación estándar",
                                            className="small mt-3",
                                        ),
                                        dbc.Checklist(
                                            id="cmp-atten-sigma-bands",
                                            options=[
                                                {
                                                    "label": "Banda ASK14 corregido",
                                                    "value": "ask14",
                                                },
                                                {
                                                    "label": "Banda NoSAm puro",
                                                    "value": "nosam",
                                                },
                                            ],
                                            value=["ask14"],
                                            switch=True,
                                        ),
                                        html.Label(
                                            "Multiplicador sigma",
                                            className="small mt-3",
                                        ),
                                        dcc.Dropdown(
                                            id="cmp-atten-sigma-multiplier",
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
                                "Comparar curva de atenuación",
                                id="cmp-atten-button",
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
                                    id="cmp-atten-graph",
                                    figure=empty_figure(
                                        "Presiona 'Comparar curva de atenuación'.",
                                        height=760,
                                    ),
                                    style={"height": "780px"},
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
                className="g-4 align-items-start mb-5",
            ),
            decomposition_card(
                prefix="cmpdecomp",
                title="Descomposición de variabilidad: ASK14 residual corregido vs NoSAm puro",
                use_external_gate_controls=True,
            ),
        ],
        fluid=True,
    )


def register_callbacks(app):
    @app.callback(
        Output("cmp-spec-summary", "children"),
        Output("cmp-spec-graph", "figure"),
        Input("cmp-spec-button", "n_clicks"),
        Input("cmp-apply-gate", "value"),
        Input("cmp-gate-alpha", "value"),
        Input("cmp-gate-beta", "value"),
        Input("cmp-spec-fixed-magnitude", "value"),
        Input("cmp-spec-fixed-rrup", "value"),
        Input("cmp-spec-fixed-zhypo", "value"),
        Input("cmp-spec-fixed-soil", "value"),
        Input("cmp-spec-fixed-rvolc", "value"),
        Input("cmp-spec-var", "value"),
        Input("cmp-spec-var-min", "value"),
        Input("cmp-spec-var-max", "value"),
        Input("cmp-spec-var-n", "value"),
        prevent_initial_call=True,
    )
    def update_multi_spectra(
        n_clicks,
        gate_values,
        gate_alpha,
        gate_beta,
        magnitude,
        rrup,
        zhypo,
        soil,
        rvolc,
        var_name,
        vmin,
        vmax,
        nvals,
    ):
        if not n_clicks:
            raise PreventUpdate

        package = get_package()

        if isinstance(package, Exception):
            return dbc.Alert(str(package), color="warning"), empty_figure(
                str(package),
                height=760,
            )

        try:
            apply_gate = "gate" in (gate_values or [])

            base_sc = scenario_dict(
                magnitude=magnitude,
                rrup=rrup,
                zhypo=zhypo,
                soil=soil,
                rvolc=rvolc,
            )

            values = varied_values(
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

            result = predict_compare_spectra(
                package=package,
                scenarios=scenarios,
                apply_gate=apply_gate,
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
            )

            fixed_values = {
                "magnitude": base_sc["magnitude"],
                "rrup_km": base_sc["rrup_km"],
                "zhypo_km": base_sc["zhypo_km"],
                "soil_class": base_sc["soil_class"],
                "rvolc_km": base_sc["rvolc_km"],
            }

            fixed_values.pop(var_name, None)

            fixed_text = (
                f"Valores fijos usados en la predicción: "
                f"Mw={fixed_values.get('magnitude', 'variable')}, "
                f"Rrup={fixed_values.get('rrup_km', 'variable')} km, "
                f"Zhypo={fixed_values.get('zhypo_km', 'variable')} km, "
                f"Soil={fixed_values.get('soil_class', 'variable')}, "
                f"Rvolc={fixed_values.get('rvolc_km', 'variable')} km"
            )

            periods = np.asarray(result["periods"], dtype=float)

            idx_001 = int(np.argmin(np.abs(periods - 0.01)))
            idx_01 = int(np.argmin(np.abs(periods - 0.1)))
            idx_10 = int(np.argmin(np.abs(periods - 1.0)))

            X0 = result.get("prediction_frame", pd.DataFrame()).head(1)

            if "Rrup_OpenQuake" in X0.columns:
                rrup_feature_txt = (
                    f" | feature ln(Rrup)="
                    f"{float(X0['Rrup_OpenQuake'].iloc[0]):.4g}"
                )
            else:
                rrup_feature_txt = ""

            diagnostic_text = (
                f"Control numérico del primer escenario: "
                f"ASK14corr Sa(0.01s)="
                f"{float(result['ask14_corrected'][0, idx_001]):.4g}, "
                f"Sa(0.1s)="
                f"{float(result['ask14_corrected'][0, idx_01]):.4g}, "
                f"Sa(1.0s)="
                f"{float(result['ask14_corrected'][0, idx_10]):.4g} | "
                f"NoSAm Sa(0.01s)="
                f"{float(result['nosam'][0, idx_001]):.4g}, "
                f"Sa(0.1s)="
                f"{float(result['nosam'][0, idx_01]):.4g}, "
                f"Sa(1.0s)="
                f"{float(result['nosam'][0, idx_10]):.4g}"
                f"{rrup_feature_txt}"
            )

            summary = [
                html.H4("Comparación múltiple", className="mb-2"),
                html.P("Modelo residual: ASK14 + GRU residual corregido"),
                html.P("Comparador: NoSAm puro"),
                html.P(gate_info_text(result["gate_obj"], apply_gate)),
                html.P(
                    f"Variable modificada: {var_name} | Escenarios: {len(scenarios)}",
                    className="mb-1",
                ),
                html.P(fixed_text, className="mb-1 text-muted small"),
                html.P(diagnostic_text, className="mb-0 text-muted small"),
            ]

            return summary, multi_spectra_figure(
                result,
                var_name,
                values,
                base_scenario=base_sc,
            )

        except Exception as exc:
            return dbc.Alert(str(exc), color="danger"), empty_figure(
                str(exc),
                height=760,
            )

    @app.callback(
        Output("cmp-spec-var-min", "value"),
        Output("cmp-spec-var-max", "value"),
        Output("cmp-spec-var-n", "value"),
        Input("cmp-spec-var", "value"),
    )
    def update_multi_spectra_range_defaults(var_name):
        defaults = {
            "magnitude": ("4.5", "7", "4"),
            "rrup_km": ("10", "400", "5"),
            "zhypo_km": ("0", "50", "5"),
            "soil_class": ("1", "5", "5"),
            "rvolc_km": ("0", "200", "5"),
        }

        return defaults.get(var_name, ("0", "1", "4"))

    @app.callback(
        Output("cmp-atten-graph", "figure"),
        Input("cmp-atten-button", "n_clicks"),
        State("cmp-apply-gate", "value"),
        State("cmp-gate-alpha", "value"),
        State("cmp-gate-beta", "value"),
        State("cmp-atten-magnitude", "value"),
        State("cmp-atten-zhypo", "value"),
        State("cmp-atten-soil", "value"),
        State("cmp-atten-rvolc", "value"),
        State("cmp-atten-period", "value"),
        State("cmp-atten-rrup-min", "value"),
        State("cmp-atten-rrup-max", "value"),
        State("cmp-atten-n-points", "value"),
        State("cmp-atten-sigma-bands", "value"),
        State("cmp-atten-sigma-multiplier", "value"),
        prevent_initial_call=True,
    )
    def update_attenuation(
        n_clicks,
        gate_values,
        gate_alpha,
        gate_beta,
        magnitude,
        zhypo,
        soil,
        rvolc,
        period,
        rrup_min,
        rrup_max,
        n_points,
        sigma_bands,
        sigma_multiplier,
    ):
        if not n_clicks:
            raise PreventUpdate

        package = get_package()

        if isinstance(package, Exception):
            return empty_figure(str(package), height=760)

        try:
            apply_gate = "gate" in (gate_values or [])

            base_sc = scenario_dict(
                magnitude=magnitude,
                rrup=rrup_min,
                zhypo=zhypo,
                soil=soil,
                rvolc=rvolc,
            )

            artifact = package["models"][MODEL_KEY_RESIDUAL]

            if period is None:
                period = float(np.asarray(artifact["periods"], dtype=float)[0])

            result = predict_compare_attenuation(
                package=package,
                base_scenario=base_sc,
                selected_period=float(period),
                apply_gate=apply_gate,
                gate_alpha=gate_alpha,
                gate_beta=gate_beta,
                n_points=n_points,
                rrup_min=rrup_min,
                rrup_max=rrup_max,
            )

            sigma_bands = sigma_bands or []

            return attenuation_compare_figure(
                package=package,
                result=result,
                show_sigma_ask14="ask14" in sigma_bands,
                show_sigma_nosam="nosam" in sigma_bands,
                sigma_multiplier=float(sigma_multiplier or 1.0),
            )

        except Exception as exc:
            return empty_figure(str(exc), height=760)

    register_decomposition_callbacks(
        app,
        prefix="cmpdecomp",
        use_external_gate_controls=True,
    )