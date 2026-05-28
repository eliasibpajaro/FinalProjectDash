from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import warnings

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


COMMON_GATE_COLS = ["Rrup_km", "Magnitude", "Hypocenter Depth (km)"]
SOIL_COL = "Soil_Class"

DEFAULT_ALPHA = 4.0
DEFAULT_BETA = 3.7184

COLORS = {
    "nosam": "#1F2937",
    "ask14": "#1D4ED8",
    "ask14_gate": "#0F766E",
    "grid": "#E5E7EB",
    "axis": "#CBD5E1",
    "text": "#111827",
    "muted": "#64748B",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_existing_file(*names):
    root = project_root()
    candidates = []

    for name in names:
        candidates.extend(
            [
                root / name,
                root.parent / name,
                Path.cwd() / name,
                Path.cwd() / "dash_espectros_residuales" / name,
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    return None


@lru_cache(maxsize=1)
def get_package():
    try:
        from model.train_model import load_model_package

        return load_model_package("model/model.pkl")
    except Exception as exc:
        return exc


@lru_cache(maxsize=1)
def load_resids_excel():
    path = find_existing_file("Resids_for_Eliasib.xlsx")

    if path is None:
        raise FileNotFoundError(
            "No encontré Resids_for_Eliasib.xlsx en la raíz del proyecto."
        )

    df = pd.read_excel(path)

    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    return df


def notebook_reference_text():
    nb_path = find_existing_file("Mixed_Effects_Eliasib.ipynb")

    if nb_path is None:
        return (
            "Notebook de referencia no encontrado: Mixed_Effects_Eliasib.ipynb. "
            "Se usa la misma lógica MixedLM implementada en Dash."
        )

    return (
        f"Notebook de referencia detectado: {nb_path.name}. "
        "No se modifica; se replica su fórmula MixedLM para graficar en Dash."
    )


def norm_name(name):
    text = str(name).strip().lower()
    for a, b in [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    ]:
        text = text.replace(a, b)

    for ch in ["(", ")", "[", "]", "/", "-", "."]:
        text = text.replace(ch, "_")

    text = "_".join(text.split())

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def pick_col(df, candidates, required=True, label="columna"):
    normalized = {norm_name(c): c for c in df.columns}

    for cand in candidates:
        key = norm_name(cand)
        if key in normalized:
            return normalized[key]

    for cand in candidates:
        key = norm_name(cand)
        for nk, real in normalized.items():
            if key and key in nk:
                return real

    if required:
        raise KeyError(
            f"No se encontró {label}. "
            f"Candidatos={candidates}. "
            f"Columnas disponibles={list(df.columns)}"
        )

    return None


def standardize_long_resids_columns(df):
    """
    Estandariza un dataframe largo sin renombrar sobre el dataframe original.
    Esto evita duplicar columnas cuando existen simultáneamente EQID_Code y EQID.
    """

    event_col = pick_col(df, ["EQID_Code", "EQID", "Event", "Event ID"], label="evento")
    station_col = pick_col(df, ["Station Code", "Station", "Station_Code", "station"], label="estación")
    period_col = pick_col(df, ["Period", "T", "Periodo"], label="periodo")
    total_col = pick_col(df, ["Total", "Residual", "residual", "Resid", "ln_residual"], label="residual Total")

    out = pd.DataFrame(
        {
            "EQID": df[event_col].astype(str),
            "Station": df[station_col].astype(str),
            "Period": pd.to_numeric(df[period_col], errors="coerce"),
            "Residual": pd.to_numeric(df[total_col], errors="coerce"),
        }
    )

    tmax_col = pick_col(df, ["Tmax", "T_max", "T max", "Tcorner"], required=False)

    if tmax_col is not None:
        tmax = pd.to_numeric(df[tmax_col], errors="coerce")
        out = out[out["Period"] <= tmax]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["EQID", "Station", "Period", "Residual"])

    return out[["EQID", "Station", "Period", "Residual"]]


@lru_cache(maxsize=1)
def nosam_long_residuals_from_excel():
    """
    Residuos NoSAm base desde Resids_for_Eliasib.xlsx.

    Replica la lógica del notebook Mixed_Effects_Eliasib.ipynb:

        pd.pivot_table(
            Resids_Eliasib,
            values="Total",
            index=["EQID_Code", "Station Code"],
            columns=["Period"]
        )

    Luego vuelve a formato largo para usar la misma función MixedLM del dashboard.
    """

    raw = load_resids_excel().copy()

    required_cols = ["EQID_Code", "Station Code", "Period", "Total"]
    missing = [c for c in required_cols if c not in raw.columns]

    if missing:
        raise KeyError(
            "NoSAm base no se puede construir porque faltan columnas en Resids_for_Eliasib.xlsx. "
            f"Faltan: {missing}. Columnas disponibles: {list(raw.columns)}"
        )

    # Igual que el notebook: una fila por evento-estación y una columna por período
    resids_per_period = pd.pivot_table(
        raw,
        values="Total",
        index=["EQID_Code", "Station Code"],
        columns=["Period"],
        aggfunc="mean",
    )

    # Volver a formato largo, eliminando períodos sin residual válido
    long_df = resids_per_period.stack(dropna=True).reset_index()
    long_df.columns = ["EQID", "Station", "Period", "Residual"]

    long_df["EQID"] = long_df["EQID"].astype(str)
    long_df["Station"] = long_df["Station"].astype(str)
    long_df["Period"] = pd.to_numeric(long_df["Period"], errors="coerce")
    long_df["Residual"] = pd.to_numeric(long_df["Residual"], errors="coerce")

    long_df = long_df.replace([np.inf, -np.inf], np.nan)
    long_df = long_df.dropna(subset=["EQID", "Station", "Period", "Residual"])

    return long_df[["EQID", "Station", "Period", "Residual"]]


def metadata_for_artifact(artifact):
    for key in ["df_wide_metadata", "df_wide", "metadata", "df_wide_preview"]:
        obj = artifact.get(key)
        if isinstance(obj, pd.DataFrame) and len(obj) > 0:
            return obj.copy()

    raise ValueError(
        "El artefacto ASK14 no contiene df_wide_metadata/df_wide_preview. "
        "No puedo mapear residuos corregidos a EQID y Station."
    )


def get_event_station_from_metadata(meta):
    event_col = pick_col(
        meta,
        ["EQID_Code", "EQID", "Event", "Event ID"],
        label="evento en metadata",
    )
    station_col = pick_col(
        meta,
        ["Station Code", "Station", "Station_Code"],
        label="estación en metadata",
    )

    return meta[event_col].astype(str).to_numpy(), meta[station_col].astype(str).to_numpy()


def ensure_physical_gate_frame(meta):
    df = meta.copy()

    mag_col = pick_col(df, ["Magnitude", "Mw", "mag"], label="Magnitude")
    zhypo_col = pick_col(
        df,
        [
            "Hypocenter Depth (km)",
            "Hypocentral Depth (km)",
            "Depth",
            "Depth_km",
            "Zhypo",
            "zhypo_km",
        ],
        label="Hypocenter Depth (km)",
    )
    rrup_col = pick_col(
        df,
        [
            "Rrup_km",
            "Rrup_OpenQuake",
            "Rrup",
            "Rrup (km)",
            "rrup_openquake",
            "rrup_km",
        ],
        label="Rrup",
    )
    soil_col = pick_col(df, ["Soil_Class", "Cat", "soil_class", "cat"], required=False)

    out = pd.DataFrame(
        {
            "Magnitude": pd.to_numeric(df[mag_col], errors="coerce"),
            "Hypocenter Depth (km)": pd.to_numeric(df[zhypo_col], errors="coerce"),
            "Rrup_km": pd.to_numeric(df[rrup_col], errors="coerce"),
        }
    )

    if soil_col is not None:
        out[SOIL_COL] = pd.to_numeric(df[soil_col], errors="coerce").fillna(1).astype(int)
    else:
        out[SOIL_COL] = 1

    return out


def subset_standard_scaler(scaler, old_cols, new_cols):
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
            "El gate guardado no contiene las variables comunes requeridas. "
            f"Faltan={missing}; gate_cont_cols={old_cols}"
        )

    if old_cols != new_cols:
        gate["scaler"] = subset_standard_scaler(gate["scaler"], old_cols, new_cols)

        if "beta" in gate and beta is None:
            gate["beta"] = float(gate["beta"]) * np.sqrt(len(new_cols) / len(old_cols))

    gate["gate_cont_cols"] = new_cols
    gate["original_gate_cont_cols"] = old_cols

    if alpha is not None:
        gate["alpha"] = float(alpha)

    if beta is not None:
        gate["beta"] = float(beta)
        gate["beta_manual"] = True

    return gate


def apply_common_gate(physical_df, gate_obj, periods):
    gate = force_common_gate_object(gate_obj)

    cols = gate["gate_cont_cols"]

    X = physical_df[cols].to_numpy(dtype=float)
    Xs = gate["scaler"].transform(X)

    d = np.linalg.norm(Xs, axis=1)

    alpha = float(gate.get("alpha", DEFAULT_ALPHA))
    beta = float(gate.get("beta", DEFAULT_BETA))

    g_dist = 1.0 / (1.0 + np.exp(alpha * (d - beta)))

    if SOIL_COL in physical_df.columns:
        soil_conf = (
            physical_df[SOIL_COL]
            .map(gate.get("soil_conf_map", {}))
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
    else:
        soil_conf = np.ones(len(physical_df), dtype=float)

    g_scalar = np.clip(g_dist * soil_conf, 0.0, 1.0)

    G = np.repeat(g_scalar[:, None], repeats=len(periods), axis=1)

    return G, g_scalar


def ask14_corrected_long_residuals(apply_gate=True, alpha=None, beta=None):
    package = get_package()

    if isinstance(package, Exception):
        raise package

    if "ask14" not in package.get("models", {}):
        raise ValueError("model.pkl no contiene el modelo 'ask14'.")

    artifact = package["models"]["ask14"]

    required = ["Y", "W", "periods", "Yhat_full"]
    missing = [k for k in required if k not in artifact]

    if missing:
        raise ValueError(f"El artefacto ASK14 no contiene estas claves: {missing}")

    Y = np.asarray(artifact["Y"], dtype=float)
    W = np.asarray(artifact["W"], dtype=float)
    Yhat = np.asarray(artifact["Yhat_full"], dtype=float)
    periods = np.asarray(artifact["periods"], dtype=float)

    if Yhat.ndim == 3 and Yhat.shape[-1] == 1:
        Yhat = Yhat[:, :, 0]

    if Y.shape != Yhat.shape:
        raise ValueError(f"Dimensiones incompatibles: Y={Y.shape}, Yhat_full={Yhat.shape}")

    if apply_gate:
        gate = artifact.get("gate")

        if gate is None:
            raise ValueError(
                "El artefacto ASK14 no tiene objeto gate; desactiva el gate para esta descomposición."
            )

        gate = force_common_gate_object(gate, alpha=alpha, beta=beta)

        meta = metadata_for_artifact(artifact)
        physical = ensure_physical_gate_frame(meta)

        G, _ = apply_common_gate(physical, gate, periods)

        eps = Y - G * Yhat
    else:
        eps = Y - Yhat

    eps = np.where(W > 0, eps, np.nan)

    meta = metadata_for_artifact(artifact)
    events, stations = get_event_station_from_metadata(meta)

    if len(events) != eps.shape[0]:
        raise ValueError(
            f"metadata tiene {len(events)} filas, pero la matriz residual tiene {eps.shape[0]} registros."
        )

    rows = []

    for j, T in enumerate(periods):
        vals = eps[:, j]
        valid = np.isfinite(vals)

        if not np.any(valid):
            continue

        rows.append(
            pd.DataFrame(
                {
                    "EQID": events[valid],
                    "Station": stations[valid],
                    "Period": float(T),
                    "Residual": vals[valid].astype(float),
                }
            )
        )

    if not rows:
        raise ValueError("No hay residuos ASK14 corregidos válidos para descomponer.")

    return pd.concat(rows, ignore_index=True)


def decompose_long_residuals(long_df, min_records_per_period=8):
    import statsmodels.formula.api as smf

    df = long_df.copy()

    df["EQID"] = df["EQID"].astype(str)
    df["Station"] = df["Station"].astype(str)
    df["Period"] = pd.to_numeric(df["Period"], errors="coerce")
    df["Residual"] = pd.to_numeric(df["Residual"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["EQID", "Station", "Period", "Residual"])

    periods = sorted(df["Period"].dropna().unique())

    records = []
    vcf = {"EQID": "0 + C(EQID)", "Station": "0 + C(Station)"}

    for T in periods:
        dreg = df.loc[
            np.isclose(df["Period"], float(T), atol=1e-10, rtol=0),
            ["EQID", "Station", "Residual"],
        ].copy()

        dreg = dreg.dropna()
        dreg["groups"] = 1

        rec = {
            "Period": float(T),
            "bias": np.nan,
            "tau": np.nan,
            "phi_s2s": np.nan,
            "phi_SS": np.nan,
            "phi": np.nan,
            "sigma": np.nan,
            "n": int(len(dreg)),
        }

        if len(dreg) < int(min_records_per_period):
            records.append(rec)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mixed = smf.mixedlm(
                    "Residual ~ 1",
                    vc_formula=vcf,
                    re_formula="0",
                    data=dreg,
                    groups="groups",
                )

                fit = mixed.fit()

            params = fit.params
            scale = float(fit.scale)

            def get_var_param(key_text):
                exact = f"{key_text} Var"

                if exact in params:
                    return float(params[exact])

                matches = [
                    k
                    for k in params.index
                    if key_text.lower() in str(k).lower()
                    and "var" in str(k).lower()
                ]

                if matches:
                    return float(params[matches[0]])

                if hasattr(fit, "vcomp") and len(fit.vcomp) >= 2:
                    if key_text.lower() == "eqid":
                        return float(fit.vcomp[0] / scale)
                    if key_text.lower() == "station":
                        return float(fit.vcomp[1] / scale)

                return np.nan

            tau = np.sqrt(max(get_var_param("EQID") * scale, 0.0))
            phi_s2s = np.sqrt(max(get_var_param("Station") * scale, 0.0))
            phi_SS = np.sqrt(max(scale, 0.0))
            phi = np.sqrt(phi_SS**2 + phi_s2s**2)
            sigma = np.sqrt(phi**2 + tau**2)

            rec.update(
                {
                    "bias": float(params.get("Intercept", np.nan)),
                    "tau": float(tau),
                    "phi_s2s": float(phi_s2s),
                    "phi_SS": float(phi_SS),
                    "phi": float(phi),
                    "sigma": float(sigma),
                }
            )

        except Exception as exc:
            rec["error"] = str(exc)

        records.append(rec)

    return pd.DataFrame(records)


_DECOMP_CACHE = {}


def decomp_cache_path():
    return project_root() / "model" / "decomp_cache.pkl"


@lru_cache(maxsize=1)
def load_precomputed_decomp_cache():
    import pickle

    path = decomp_cache_path()

    if not path.exists():
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def _gate_cache_records():
    """
    Lee las combinaciones alpha/beta disponibles en model/decomp_cache.pkl.

    Ejemplo de clave:
    ask14_corrected__gate1__alpha_3.0000__beta_2.7845
    """
    import re

    cache = load_precomputed_decomp_cache()

    if cache is None:
        return []

    items = cache.get("items", {})

    records = []
    pattern = re.compile(
        r"^ask14_corrected__gate1__alpha_([0-9.+-eE]+)__beta_([0-9.+-eE]+)$"
    )

    for key in items.keys():
        match = pattern.match(str(key))

        if match:
            records.append(
                {
                    "key": key,
                    "alpha": round(float(match.group(1)), 4),
                    "beta": round(float(match.group(2)), 4),
                }
            )

    records = sorted(records, key=lambda x: (x["beta"], x["alpha"]))

    return records


def _unique_sorted(values):
    return sorted(list(dict.fromkeys([round(float(v), 4) for v in values])))


def _nearest_value(value, candidates):
    candidates = _unique_sorted(candidates)

    if not candidates:
        return float(value)

    value = float(value)

    return min(candidates, key=lambda x: abs(float(x) - value))


def gate_beta_values():
    records = _gate_cache_records()

    if not records:
        return [DEFAULT_BETA]

    return _unique_sorted([r["beta"] for r in records])


def gate_alpha_values(beta=None):
    records = _gate_cache_records()

    if not records:
        return [DEFAULT_ALPHA]

    if beta is None:
        return _unique_sorted([r["alpha"] for r in records])

    beta = _nearest_value(beta, [r["beta"] for r in records])

    vals = [
        r["alpha"]
        for r in records
        if abs(float(r["beta"]) - float(beta)) <= 5e-4
    ]

    if not vals:
        return _unique_sorted([r["alpha"] for r in records])

    return _unique_sorted(vals)


def gate_alpha_dropdown_options(beta=None):
    return [
        {"label": f"{float(v):g}", "value": float(v)}
        for v in gate_alpha_values(beta=beta)
    ]


def gate_beta_dropdown_options():
    return [
        {"label": f"{float(v):.4f}", "value": float(v)}
        for v in gate_beta_values()
    ]


def default_cached_gate_params(alpha_default=DEFAULT_ALPHA, beta_default=DEFAULT_BETA):
    records = _gate_cache_records()

    if not records:
        return float(alpha_default), float(beta_default)

    betas = gate_beta_values()
    beta_used = _nearest_value(beta_default, betas)

    alphas = gate_alpha_values(beta=beta_used)
    alpha_used = _nearest_value(alpha_default, alphas)

    return float(alpha_used), float(beta_used)


def resolve_precomputed_gate_params(alpha=None, beta=None):
    """
    Fuerza alpha/beta a una combinación realmente existente en decomp_cache.pkl.

    Esto evita que Dash pida alpha=3.0000000001 o beta con otro redondeo
    y termine leyendo una combinación equivocada o inexistente.
    """
    records = _gate_cache_records()

    if not records:
        alpha_used = DEFAULT_ALPHA if alpha in [None, ""] else float(alpha)
        beta_used = DEFAULT_BETA if beta in [None, ""] else float(beta)
        return float(alpha_used), float(beta_used)

    beta_input = DEFAULT_BETA if beta in [None, ""] else float(beta)
    beta_used = _nearest_value(beta_input, [r["beta"] for r in records])

    available_alphas = [
        r["alpha"]
        for r in records
        if abs(float(r["beta"]) - float(beta_used)) <= 5e-4
    ]

    alpha_input = DEFAULT_ALPHA if alpha in [None, ""] else float(alpha)
    alpha_used = _nearest_value(alpha_input, available_alphas)

    return float(alpha_used), float(beta_used)


def make_precomputed_key(model_name, apply_gate=True, alpha=None, beta=None):
    if model_name == "nosam":
        return "nosam"

    if model_name == "ask14_corrected" and not apply_gate:
        return "ask14_corrected__gate0"

    if model_name == "ask14_corrected" and apply_gate:
        if alpha is None or beta is None:
            raise ValueError(
                "Para leer ASK14 con gate desde cache debes indicar alpha y beta."
            )

        return (
            "ask14_corrected__gate1__"
            f"alpha_{float(alpha):.4f}__"
            f"beta_{float(beta):.4f}"
        )

    raise ValueError(f"Combinación no reconocida: {model_name}, gate={apply_gate}")


def cached_decomp(model_name, apply_gate=True, alpha=None, beta=None):
    """
    Lee la descomposición desde model/decomp_cache.pkl.

    Si model_name='ask14_corrected' y apply_gate=True, alpha/beta se ajustan
    a la combinación precalculada más cercana.
    """
    if model_name == "ask14_corrected" and apply_gate:
        alpha, beta = resolve_precomputed_gate_params(alpha=alpha, beta=beta)

    alpha_key = None if alpha is None else round(float(alpha), 4)
    beta_key = None if beta is None else round(float(beta), 4)

    memory_key = (model_name, bool(apply_gate), alpha_key, beta_key)

    if memory_key in _DECOMP_CACHE:
        return _DECOMP_CACHE[memory_key].copy()

    cache = load_precomputed_decomp_cache()

    if cache is None:
        raise FileNotFoundError(
            "No existe model/decomp_cache.pkl. "
            "Ejecuta primero: python model/precompute_decomp.py --alphas \"1:6:0.5\""
        )

    key = make_precomputed_key(
        model_name=model_name,
        apply_gate=apply_gate,
        alpha=alpha,
        beta=beta,
    )

    items = cache.get("items", {})

    if key not in items:
        available = "\n".join(items.keys())

        raise KeyError(
            "No existe esa descomposición precalculada en model/decomp_cache.pkl.\n\n"
            f"Clave buscada: {key}\n\n"
            f"Claves disponibles:\n{available}\n\n"
            "Si cambiaste alpha/beta, primero precalcula esa combinación con:\n"
            f"python model/precompute_decomp.py --alpha {alpha} --beta {beta}"
        )

    out = items[key].copy()
    _DECOMP_CACHE[memory_key] = out.copy()

    return out

def decomp_figure(apply_gate=True, alpha=None, beta=None):
    if apply_gate:
        alpha, beta = resolve_precomputed_gate_params(alpha=alpha, beta=beta)

    nosam = cached_decomp("nosam", apply_gate=False)

    ask14 = cached_decomp(
        "ask14_corrected",
        apply_gate=apply_gate,
        alpha=alpha,
        beta=beta,
    )

    if apply_gate:
        gate_label = f"con gate | α={alpha:g}, β={beta:.4f}"
    else:
        gate_label = "sin gate"

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "τ — Between-event",
            "ϕ — Within-event + station",
            "σ — Total",
        ],
        horizontal_spacing=0.08,
    )

    for col_idx, metric in enumerate(["tau", "phi", "sigma"], start=1):
        fig.add_trace(
            go.Scatter(
                x=nosam["Period"],
                y=nosam[metric],
                mode="lines+markers",
                name="NoSAm base",
                legendgroup="nosam",
                showlegend=(col_idx == 1),
                line=dict(color=COLORS["nosam"], width=2.8, dash="dash"),
                marker=dict(
                    size=7,
                    color=COLORS["nosam"],
                    symbol="circle-open",
                    line=dict(width=1.6),
                ),
                hovertemplate=f"T=%{{x:g}} s<br>{metric}=%{{y:.4f}}<extra>NoSAm</extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.add_trace(
            go.Scatter(
                x=ask14["Period"],
                y=ask14[metric],
                mode="lines+markers",
                name=f"ASK14 + GRU residual ({gate_label})",
                legendgroup="ask14",
                showlegend=(col_idx == 1),
                line=dict(
                    color=COLORS["ask14_gate"] if apply_gate else COLORS["ask14"],
                    width=3.1,
                ),
                marker=dict(
                    size=7,
                    color=COLORS["ask14_gate"] if apply_gate else COLORS["ask14"],
                ),
                hovertemplate=f"T=%{{x:g}} s<br>{metric}=%{{y:.4f}}<extra>ASK14 corregido</extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            type="log",
            title_text="Período T (s)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["axis"],
            row=1,
            col=col_idx,
        )

        fig.update_yaxes(
            title_text="Desviación estándar" if col_idx == 1 else None,
            showgrid=True,
            gridcolor=COLORS["grid"],
            linecolor=COLORS["axis"],
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Descomposición de variabilidad: τ, ϕ y σ — {gate_label}",
        height=560,
        margin=dict(l=50, r=30, t=85, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Inter, Segoe UI, Arial, sans-serif",
            size=13,
            color=COLORS["text"],
        ),
        title_font=dict(size=18, color=COLORS["text"]),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
    )

    return fig


def decomp_summary(apply_gate=True, alpha=None, beta=None):
    alpha_requested = alpha
    beta_requested = beta

    if apply_gate:
        alpha, beta = resolve_precomputed_gate_params(alpha=alpha, beta=beta)

    nosam = cached_decomp("nosam", apply_gate=False)

    ask14 = cached_decomp(
        "ask14_corrected",
        apply_gate=apply_gate,
        alpha=alpha,
        beta=beta,
    )

    merged = nosam[["Period", "tau", "phi", "sigma"]].merge(
        ask14[["Period", "tau", "phi", "sigma"]],
        on="Period",
        suffixes=("_nosam", "_ask14"),
    )

    if apply_gate:
        cache_key = make_precomputed_key(
            "ask14_corrected",
            apply_gate=True,
            alpha=alpha,
            beta=beta,
        )
    else:
        cache_key = "ask14_corrected__gate0"

    cards = []

    for metric, label in [("tau", "τ"), ("phi", "ϕ"), ("sigma", "σ")]:
        base = merged[f"{metric}_nosam"].to_numpy(dtype=float)
        corr = merged[f"{metric}_ask14"].to_numpy(dtype=float)

        valid = np.isfinite(base) & np.isfinite(corr) & (base > 0)

        if valid.any():
            red = 100.0 * (base[valid] - corr[valid]) / base[valid]
            txt = f"{np.nanmean(red):.2f}%"
        else:
            txt = "NA"

        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(f"Reducción media {label}", className="text-muted small"),
                            html.H4(txt, className="mb-0"),
                        ]
                    ),
                    className="soft-card h-100",
                ),
                md=4,
            )
        )

    return html.Div(
        [
            dbc.Row(cards, className="g-3 mb-2"),
            dbc.Alert(
                [
                    html.Strong("Descomposición leída: "),
                    html.Span(cache_key),
                    html.Br(),
                    html.Small(
                        f"Gate: {'activado' if apply_gate else 'desactivado'} | "
                        f"alpha solicitado={alpha_requested} | beta solicitado={beta_requested} | "
                        f"alpha usado={alpha} | beta usado={beta}"
                    ),
                ],
                color="light",
                className="border mb-0",
            ),
        ]
    )


def error_figure(message):
    fig = go.Figure()

    fig.add_annotation(
        text=str(message),
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["muted"]),
    )

    fig.update_layout(
        template="plotly_white",
        height=560,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


def decomposition_card(
    prefix,
    title="Descomposición de variabilidad",
    use_external_gate_controls=False,
):
    controls = []

    if not use_external_gate_controls:
        controls = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Gate en ASK14 corregido"),
                            dbc.Checklist(
                                id=f"{prefix}-apply-gate",
                                options=[{"label": "Aplicar gate", "value": "gate"}],
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
                            dbc.Input(
                                id=f"{prefix}-alpha",
                                type="number",
                                value=DEFAULT_ALPHA,
                                step=0.0001,
                                min=0.01,
                                disabled=True,
                            ),
                        ],
                        xs=12,
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Beta"),
                            dbc.Input(
                                id=f"{prefix}-beta",
                                type="number",
                                value=DEFAULT_BETA,
                                step=0.0001,
                                min=0.01,
                                disabled=True,
                            ),
                        ],
                        xs=12,
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Actualizar"),
                            dbc.Button(
                                "Actualizar decomp",
                                id=f"{prefix}-button",
                                color="dark",
                                className="w-100",
                            ),
                        ],
                        xs=12,
                        md=2,
                    ),
                ],
                className="g-3 mb-3",
            )
        ]
    else:
        controls = [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Alert(
                            "Esta descomposición usa el switch global de gate, alpha y beta de esta pestaña.",
                            color="light",
                            className="border mb-0",
                        ),
                        md=9,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Actualizar decomp",
                            id=f"{prefix}-button",
                            color="dark",
                            className="w-100",
                        ),
                        md=3,
                    ),
                ],
                className="g-3 mb-3 align-items-center",
            )
        ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="mb-2"),
                html.P(
                    "Se replica la descomposición mixed-effects del notebook: intercepto fijo, "
                    "efecto aleatorio por evento y efecto aleatorio por estación. "
                    "NoSAm usa el residual Total de Resids_for_Eliasib.xlsx; ASK14 usa el residual remanente "
                    "después de aplicar la corrección GRU.",
                    className="text-muted",
                ),
                dbc.Alert(notebook_reference_text(), color="light", className="border"),
                *controls,
                html.Div(id=f"{prefix}-summary", className="mb-3"),
                dcc.Loading(
                    type="circle",
                    children=dcc.Graph(
                        id=f"{prefix}-graph",
                        style={"height": "590px"},
                        config={"responsive": True},
                    ),
                ),
            ]
        ),
        className="soft-card mb-4",
    )


def register_decomposition_callbacks(app, prefix, use_external_gate_controls=False):
    if use_external_gate_controls:

        @app.callback(
            Output(f"{prefix}-summary", "children"),
            Output(f"{prefix}-graph", "figure"),
            Input(f"{prefix}-button", "n_clicks"),
            Input("cmp-apply-gate", "value"),
            Input("cmp-gate-alpha", "value"),
            Input("cmp-gate-beta", "value"),
        )
        def _update_decomp_external(n_clicks, gate_values, alpha, beta):
            """
            Para la pestaña ASK14 vs NoSAm.

            Importante:
            alpha y beta son Input, no State.
            Por tanto, cambiar alpha arriba debe actualizar la descomposición.
            """
            apply_gate = "gate" in (gate_values or [])

            try:
                alpha = DEFAULT_ALPHA if alpha in [None, ""] else float(alpha)
                beta = DEFAULT_BETA if beta in [None, ""] else float(beta)

                return (
                    decomp_summary(apply_gate=apply_gate, alpha=alpha, beta=beta),
                    decomp_figure(apply_gate=apply_gate, alpha=alpha, beta=beta),
                )

            except Exception as exc:
                return dbc.Alert(str(exc), color="danger"), error_figure(str(exc))

    else:

        @app.callback(
            Output(f"{prefix}-summary", "children"),
            Output(f"{prefix}-graph", "figure"),
            Input(f"{prefix}-button", "n_clicks"),
            Input(f"{prefix}-apply-gate", "value"),
            Input(f"{prefix}-alpha", "value"),
            Input(f"{prefix}-beta", "value"),
        )
        def _update_decomp_internal(n_clicks, gate_values, alpha, beta):
            """
            Para la pestaña Resultados.

            También se actualiza cuando cambias alpha/beta en la propia tarjeta.
            """
            apply_gate = "gate" in (gate_values or [])

            try:
                alpha = DEFAULT_ALPHA if alpha in [None, ""] else float(alpha)
                beta = DEFAULT_BETA if beta in [None, ""] else float(beta)

                return (
                    decomp_summary(apply_gate=apply_gate, alpha=alpha, beta=beta),
                    decomp_figure(apply_gate=apply_gate, alpha=alpha, beta=beta),
                )

            except Exception as exc:
                return dbc.Alert(str(exc), color="danger"), error_figure(str(exc))