# Código completo del proyecto


## `app.py`

```python
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from tabs import introduccion, problema, objetivos, resultados, prediccion, limitaciones, conclusiones

external_stylesheets = [
    dbc.themes.FLATLY,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Predicción de Espectros de Respuesta",
)

server = app.server

TAB_MODULES = {
    "introduccion": introduccion,
    "problema": problema,
    "objetivos": objetivos,
    "resultados": resultados,
    "prediccion": prediccion,
    "limitaciones": limitaciones,
    "conclusiones": conclusiones,
}


def make_navbar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div("Sa", className="brand-badge"), width="auto"),
                        dbc.Col(
                            [
                                html.Div("Dashboard de Aceleraciones Espectrales", className="brand-title"),
                                html.Div("Aprendizaje residual + GRU + validación LOGO", className="brand-subtitle"),
                            ],
                            width=True,
                        ),
                    ],
                    align="center",
                    className="g-2",
                )
            ],
            fluid=True,
        ),
        className="top-navbar",
        dark=False,
    )


def make_tabs():
    return dbc.Card(
        dbc.CardBody(
            dcc.Tabs(
                id="main-tabs",
                value="introduccion",
                className="custom-tabs",
                children=[
                    dcc.Tab(label="Introducción", value="introduccion", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Problema", value="problema", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Objetivos", value="objetivos", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Resultados", value="resultados", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Predicción", value="prediccion", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Limitaciones", value="limitaciones", className="custom-tab", selected_className="custom-tab-selected"),
                    dcc.Tab(label="Conclusiones", value="conclusiones", className="custom-tab", selected_className="custom-tab-selected"),
                ],
            )
        ),
        className="tabs-card",
    )


app.layout = html.Div(
    [
        make_navbar(),
        dbc.Container(
            [
                make_tabs(),
                html.Div(id="tab-content", className="tab-content-wrapper"),
            ],
            fluid=True,
            className="main-container",
        ),
    ]
)


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab_name):
    module = TAB_MODULES.get(tab_name, introduccion)
    return module.layout()


# Registro desacoplado de callbacks por módulo.
for module in TAB_MODULES.values():
    if hasattr(module, "register_callbacks"):
        module.register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
```


## `model/train_model.py`

```python
"""
Entrenamiento de modelos residuales GRU para espectros de respuesta.

Modelos soportados:
1. nosam: residual NoSAm + variables base.
2. nosam_elevation: residual NoSAm + variables base + log(Station Elevation).
3. ask14: residual ASK14 calculado desde Sa observado y OpenQuake, si está instalado.

La salida se guarda en model/model.pkl como un paquete serializado con:
- modelos Keras empaquetados en JSON + pesos
- preprocesadores sklearn
- períodos
- métricas LOGO y full train
- metadatos por modelo
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers

# =========================================================
# CONFIGURACIÓN GLOBAL
# =========================================================

SEED = 42

EVENT_COL = "EQID_Code"
STATION_COL = "Station Code"
RECORD_COL = "Record Sequence Number"

SOIL_COL = "Soil_Class"
TMAX_COL = "Tmax"
PERIOD_COL = "Period"

SA_COL = "Sa"
NOSAM_RESIDUAL_COL = "Total"
TARGET_COL = "TargetResidual"

RVOLC_COL = "Rvolc [km]"
ELEV_COL = "Station Elevation (m)"
LOG_ELEV_COL = "log_Station_Elevation"

RRUP_RAW_COL = "Rrup_raw_km"
RVOLC_RAW_COL = "Rvolc_raw_km"

BASE_CONT_COLS_NOSAM = [
    "Hypocenter Depth (km)",
    "Magnitude",
    "Rrup_OpenQuake",
    RVOLC_COL,
]
BASE_CONT_COLS_NOSAM_ELEV = BASE_CONT_COLS_NOSAM + [LOG_ELEV_COL]
BASE_CONT_COLS_ASK14 = [
    "Hypocenter Depth (km)",
    "Magnitude",
    "Rrup_OpenQuake",
]
CAT_COLS = [SOIL_COL]

MODEL_CONFIGS = {
    "nosam": {
        "label": "NoSAm + GRU",
        "target_source": "nosam_residual",
        "continuous_cols": BASE_CONT_COLS_NOSAM,
        "cat_cols": CAT_COLS,
        "needs_elevation": False,
        "needs_ask14": False,
    },
    "nosam_elevation": {
        "label": "NoSAm + GRU + Elevación",
        "target_source": "nosam_residual",
        "continuous_cols": BASE_CONT_COLS_NOSAM_ELEV,
        "cat_cols": CAT_COLS,
        "needs_elevation": True,
        "needs_ask14": False,
    },
    "ask14": {
        "label": "ASK14 + GRU",
        "target_source": "ask14_residual",
        "continuous_cols": BASE_CONT_COLS_ASK14,
        "cat_cols": CAT_COLS,
        "needs_elevation": False,
        "needs_ask14": True,
    },
}


# =========================================================
# REPRODUCIBILIDAD
# =========================================================

def reset_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# =========================================================
# CARGA Y PREPARACIÓN DE DATOS
# =========================================================

def resolve_path(path_like: str | Path) -> Path:
    """Busca un archivo en la ruta dada, raíz del proyecto, ./data y ./model."""
    p = Path(path_like)
    if p.exists():
        return p

    candidates = [
        Path.cwd() / p,
        Path.cwd() / "data" / p.name,
        Path.cwd() / "model" / p.name,
        Path(__file__).resolve().parents[1] / p.name,
        Path(__file__).resolve().parents[1] / "data" / p.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No se encontró el archivo {path_like}. "
        "Ubícalo en la raíz del proyecto o define la ruta completa."
    )


def read_table(path_like: str | Path) -> pd.DataFrame:
    path = resolve_path(path_like)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)

    raise ValueError(f"Formato no soportado: {path.suffix}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas para que todos los notebooks usen una convención única."""
    df = df.copy()

    rename_map = {}

    if "Cat" in df.columns and SOIL_COL not in df.columns:
        rename_map["Cat"] = SOIL_COL

    if "Tcorner" in df.columns and TMAX_COL not in df.columns:
        rename_map["Tcorner"] = TMAX_COL

    if "T_max" in df.columns and TMAX_COL not in df.columns:
        rename_map["T_max"] = TMAX_COL

    if "Tmax" in df.columns:
        rename_map["Tmax"] = TMAX_COL

    if RVOLC_COL not in df.columns:
        for candidate in ["Rvolc", "Rvolc_km", "Rvolc_km_", "Rvolc (km)", "Rvolc[km]"]:
            if candidate in df.columns:
                rename_map[candidate] = RVOLC_COL
                break

    df = df.rename(columns=rename_map)

    return df


def merge_station_elevation(df: pd.DataFrame, meta_path: Optional[str | Path]) -> pd.DataFrame:
    """
    Agrega Station Elevation desde CopiaDataBaseSGC2 si no está en la base larga.
    Prioridad de merge:
    1. Record Sequence Number
    2. Station Code
    """
    df = df.copy()

    if ELEV_COL in df.columns:
        return df

    if meta_path is None:
        df[ELEV_COL] = np.nan
        return df

    df_meta = read_table(meta_path)
    df_meta = standardize_columns(df_meta)

    if ELEV_COL not in df_meta.columns:
        df[ELEV_COL] = np.nan
        return df

    if RECORD_COL in df.columns and RECORD_COL in df_meta.columns:
        cols = [RECORD_COL, ELEV_COL]
        df = df.merge(df_meta[cols].drop_duplicates(subset=[RECORD_COL]), on=RECORD_COL, how="left")
        return df

    if STATION_COL in df.columns and STATION_COL in df_meta.columns:
        cols = [STATION_COL, ELEV_COL]
        df = df.merge(df_meta[cols].drop_duplicates(subset=[STATION_COL]), on=STATION_COL, how="left")
        return df

    df[ELEV_COL] = np.nan
    return df


def load_long_dataset(
    data_path: str | Path,
    meta_path: Optional[str | Path] = None,
    model_key: str = "nosam",
) -> pd.DataFrame:
    """
    Carga Resids_for_Eliasib en formato largo.
    Conserva Rrup y Rvolc originales para predicción física y crea columnas transformadas:
    - Rrup_OpenQuake = log(Rrup_raw_km)
    - Rvolc [km] = log1p(Rvolc_raw_km), solo para modelos NoSAm
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Modelo no reconocido: {model_key}")

    cfg = MODEL_CONFIGS[model_key]

    df = read_table(data_path)
    df = standardize_columns(df)

    required = [
        RECORD_COL,
        EVENT_COL,
        STATION_COL,
        "Hypocenter Depth (km)",
        "Magnitude",
        "Rrup_OpenQuake",
        SOIL_COL,
        TMAX_COL,
        PERIOD_COL,
        SA_COL,
    ]

    if cfg["target_source"] == "nosam_residual":
        required.append(NOSAM_RESIDUAL_COL)

    if model_key.startswith("nosam"):
        required.append(RVOLC_COL)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para {model_key}: {missing}")

    numeric_cols = [
        "Hypocenter Depth (km)",
        "Magnitude",
        "Rrup_OpenQuake",
        TMAX_COL,
        PERIOD_COL,
        SA_COL,
        SOIL_COL,
    ]

    if RVOLC_COL in df.columns:
        numeric_cols.append(RVOLC_COL)

    if NOSAM_RESIDUAL_COL in df.columns:
        numeric_cols.append(NOSAM_RESIDUAL_COL)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[RRUP_RAW_COL] = df["Rrup_OpenQuake"].astype(float)
    if (df[RRUP_RAW_COL] <= 0).any():
        n_bad = int((df[RRUP_RAW_COL] <= 0).sum())
        raise ValueError(f"Hay {n_bad} filas con Rrup_OpenQuake <= 0; no se puede aplicar log.")

    df["Rrup_OpenQuake"] = np.log(df[RRUP_RAW_COL])

    if RVOLC_COL in df.columns:
        df[RVOLC_RAW_COL] = df[RVOLC_COL].astype(float)
        if (df[RVOLC_RAW_COL] < 0).any():
            n_bad = int((df[RVOLC_RAW_COL] < 0).sum())
            raise ValueError(f"Hay {n_bad} filas con {RVOLC_COL} < 0.")
        df[RVOLC_COL] = np.log1p(df[RVOLC_RAW_COL])

    if cfg["needs_elevation"]:
        df = merge_station_elevation(df, meta_path)
        df[ELEV_COL] = pd.to_numeric(df[ELEV_COL], errors="coerce")
        median_elev = df[ELEV_COL].median()
        df[ELEV_COL] = df[ELEV_COL].fillna(median_elev)
        df[LOG_ELEV_COL] = np.log(df[ELEV_COL].clip(lower=1e-6))

    if cfg["target_source"] == "nosam_residual":
        df[TARGET_COL] = df[NOSAM_RESIDUAL_COL]
    elif cfg["target_source"] == "ask14_residual":
        df = add_ask14_residual(df)
    else:
        raise ValueError(f"target_source no reconocido: {cfg['target_source']}")

    keep_cols = [
        RECORD_COL,
        EVENT_COL,
        STATION_COL,
        "Hypocenter Depth (km)",
        "Magnitude",
        "Rrup_OpenQuake",
        RRUP_RAW_COL,
        SOIL_COL,
        TMAX_COL,
        PERIOD_COL,
        SA_COL,
        TARGET_COL,
    ]

    if RVOLC_COL in df.columns:
        keep_cols += [RVOLC_COL, RVOLC_RAW_COL]

    if cfg["needs_elevation"]:
        keep_cols += [ELEV_COL, LOG_ELEV_COL]

    df = df.dropna(subset=[c for c in keep_cols if c in df.columns]).reset_index(drop=True)

    return df


def build_wide_dataset(df_long: pd.DataFrame, model_key: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte el dataset largo a formato ancho por registro.

    Retorna:
    - df_wide
    - periods
    - Y_raw: matriz de residuos objetivo
    - W: máscara final por observación + Tmax
    - Y: matriz de entrenamiento con ceros donde W=0
    - Sa_raw: matriz de Sa observado lineal
    """
    cfg = MODEL_CONFIGS[model_key]
    periods = np.array(sorted(df_long[PERIOD_COL].dropna().unique()), dtype=float)

    index_cols = [
        RECORD_COL,
        EVENT_COL,
        STATION_COL,
        "Hypocenter Depth (km)",
        "Magnitude",
        "Rrup_OpenQuake",
        RRUP_RAW_COL,
        SOIL_COL,
        TMAX_COL,
    ]

    if RVOLC_COL in df_long.columns:
        index_cols += [RVOLC_COL, RVOLC_RAW_COL]

    if cfg["needs_elevation"]:
        index_cols += [ELEV_COL, LOG_ELEV_COL]

    dup_check = (
        df_long[index_cols]
        .drop_duplicates()
        .groupby(RECORD_COL)
        .size()
    )

    if (dup_check > 1).any():
        bad_records = dup_check[dup_check > 1].index.tolist()[:10]
        raise ValueError(f"Records con metadatos inconsistentes. Ejemplos: {bad_records}")

    df_target = (
        df_long.pivot_table(
            index=index_cols,
            columns=PERIOD_COL,
            values=TARGET_COL,
            aggfunc="first",
        )
        .reset_index()
    )

    df_sa = (
        df_long.pivot_table(
            index=[RECORD_COL],
            columns=PERIOD_COL,
            values=SA_COL,
            aggfunc="first",
        )
        .reset_index()
    )

    period_cols_sorted = [p for p in periods if p in df_target.columns]

    # df_target y df_sa pueden quedar con órdenes distintos; por eso se alinea por RECORD_COL.
    sa_cols_renamed = {p: f"Sa_period_{p}" for p in period_cols_sorted}
    df_sa = df_sa[[RECORD_COL] + period_cols_sorted].rename(columns=sa_cols_renamed)

    df_wide = df_target[index_cols + period_cols_sorted].copy()
    df_wide = df_wide.merge(df_sa, on=RECORD_COL, how="left")

    Y_raw = df_wide[period_cols_sorted].to_numpy(dtype=float)
    Sa_raw = df_wide[[sa_cols_renamed[p] for p in period_cols_sorted]].to_numpy(dtype=float)

    # Mantener df_wide limpio para preprocesamiento y métricas.
    df_wide = df_wide[index_cols + period_cols_sorted].copy()

    W_obs = np.isfinite(Y_raw)
    tmax = df_wide[TMAX_COL].to_numpy(dtype=float).reshape(-1, 1)
    W_tmax = periods.reshape(1, -1) <= tmax

    W = (W_obs & W_tmax).astype(np.float32)
    Y = np.where(W == 1.0, np.nan_to_num(Y_raw, nan=0.0), 0.0).astype(np.float32)

    return df_wide, periods, Y_raw, W, Y, Sa_raw


# =========================================================
# ASK14 OPCIONAL
# =========================================================

def soil_class_to_vs30(soil: int) -> float:
    mapping = {1: 1600.0, 2: 800.0, 3: 450.0, 4: 250.0, 5: 100.0}
    return float(mapping.get(int(soil), 450.0))


def calculate_z1pt0(vs30):
    vs30 = np.asarray(vs30, dtype=float)
    c1 = 571.0 ** 4
    c2 = 1360.0 ** 4
    return np.exp((-7.15 / 4.0) * np.log((vs30 ** 4 + c1) / (c2 + c1)))


def calculate_z2pt5_ngaw2(vs30):
    vs30 = np.asarray(vs30, dtype=float)
    return np.exp(7.089 - 1.144 * np.log(vs30))


def estimate_width_ask14(dip, mag):
    dip = np.asarray(dip, dtype=float)
    mag = np.asarray(mag, dtype=float)
    return np.minimum(18.0 / np.sin(np.radians(dip)), 10.0 ** (-1.75 + 0.45 * mag))


def _first_float(x) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else np.nan


def ask14_ln_spectrum(periods, Mag, Rrup, Zhypo, Vs30_Val, dip=90.0, rake=-15.03) -> np.ndarray:
    """
    Calcula ln(Sa) con ASK14 usando OpenQuake.
    Requiere instalar openquake.engine.
    """
    try:
        from openquake.hazardlib import const
        from openquake.hazardlib.contexts import DistancesContext, RuptureContext, SitesContext
        from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014 as ASK14
        from openquake.hazardlib.imt import SA
    except Exception as exc:
        raise ImportError(
            "Para entrenar o predecir el modelo ASK14 instala OpenQuake: "
            "pip install openquake.engine"
        ) from exc

    periods = np.asarray(periods, dtype=float)

    rup_ctx = RuptureContext()
    rup_ctx.mag = float(Mag)
    rup_ctx.width = float(estimate_width_ask14(dip, Mag))
    rup_ctx.ztor = 0.0
    rup_ctx.hypo_depth = float(Zhypo)
    rup_ctx.rake = float(rake)
    rup_ctx.dip = float(dip)

    site_ctx = SitesContext()
    site_ctx.vs30 = np.array([float(Vs30_Val)], dtype=float)
    site_ctx.vs30measured = np.array([False], dtype=bool)
    site_ctx.z1pt0 = np.array([_first_float(calculate_z1pt0(Vs30_Val))], dtype=float)
    site_ctx.z2pt5 = np.array([_first_float(calculate_z2pt5_ngaw2(Vs30_Val))], dtype=float)
    site_ctx.sids = np.array([0], dtype=int)
    site_ctx.backarc = np.array([False], dtype=bool)

    dist_ctx = DistancesContext()
    dist_ctx.rhypo = np.array([float(Rrup)], dtype=float)
    dist_ctx.rrup = np.array([float(Rrup)], dtype=float)
    dist_ctx.rjb = np.array([float(Rrup)], dtype=float)
    dist_ctx.repi = np.array([float(Rrup)], dtype=float)
    dist_ctx.rx = np.array([0.0], dtype=float)
    dist_ctx.ry0 = np.array([0.0], dtype=float)

    mean_ln = np.empty(len(periods), dtype=float)
    gmm = ASK14()

    for j, t in enumerate(periods):
        try:
            mean, _ = gmm.get_mean_and_stddevs(
                site_ctx,
                rup_ctx,
                dist_ctx,
                imt=SA(float(t)),
                stddev_types=[const.StdDev.TOTAL],
            )
            mean_ln[j] = _first_float(mean)
        except Exception:
            mean_ln[j] = np.nan

    return mean_ln


def add_ask14_residual(df_long: pd.DataFrame) -> pd.DataFrame:
    """Crea TargetResidual = ln(Sa_obs) - ln(Sa_ASK14)."""
    df = df_long.copy()

    required = [RECORD_COL, PERIOD_COL, SA_COL, "Magnitude", RRUP_RAW_COL, "Hypocenter Depth (km)", SOIL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"No se puede calcular ASK14; faltan columnas: {missing}")

    periods = np.array(sorted(df[PERIOD_COL].dropna().unique()), dtype=float)

    record_meta = (
        df[[RECORD_COL, "Magnitude", RRUP_RAW_COL, "Hypocenter Depth (km)", SOIL_COL]]
        .drop_duplicates(subset=[RECORD_COL])
        .reset_index(drop=True)
    )

    rows = []
    for _, row in record_meta.iterrows():
        vs30 = soil_class_to_vs30(int(row[SOIL_COL]))
        ln_ask = ask14_ln_spectrum(
            periods=periods,
            Mag=float(row["Magnitude"]),
            Rrup=float(row[RRUP_RAW_COL]),
            Zhypo=float(row["Hypocenter Depth (km)"]),
            Vs30_Val=vs30,
        )
        for period, ln_val in zip(periods, ln_ask):
            rows.append({RECORD_COL: row[RECORD_COL], PERIOD_COL: period, "lnSa_ASK14": ln_val})

    ask_df = pd.DataFrame(rows)
    df = df.merge(ask_df, on=[RECORD_COL, PERIOD_COL], how="left")
    df[TARGET_COL] = np.log(df[SA_COL].astype(float).clip(lower=1e-12)) - df["lnSa_ASK14"]

    return df


# =========================================================
# MODELO GRU RESIDUAL
# =========================================================

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(model_key: str) -> ColumnTransformer:
    cfg = MODEL_CONFIGS[model_key]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg["continuous_cols"]),
            ("cat", make_ohe(), cfg["cat_cols"]),
        ],
        remainder="drop",
    )


def build_sequence_input(Xp: np.ndarray, periods: np.ndarray) -> np.ndarray:
    n = Xp.shape[0]
    p = len(periods)

    X_rep = np.repeat(Xp[:, None, :], repeats=p, axis=1)
    log_t = np.log(periods).reshape(1, p, 1)
    log_t = np.repeat(log_t, repeats=n, axis=0)

    return np.concatenate([X_rep, log_t], axis=2).astype(np.float32)


def get_feature_matrix(df_wide: pd.DataFrame, model_key: str) -> pd.DataFrame:
    cfg = MODEL_CONFIGS[model_key]
    feature_cols = cfg["continuous_cols"] + cfg["cat_cols"]
    missing = [c for c in feature_cols if c not in df_wide.columns]
    if missing:
        raise ValueError(f"Faltan columnas predictoras para {model_key}: {missing}")

    return df_wide[feature_cols].copy()


def build_residual_model(seq_len: int, feat_dim: int, seed: int = SEED) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len, feat_dim), name="seq_input")

    x = layers.GRU(
        units=8,
        return_sequences=True,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed + 1),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="gru_base",
    )(inp)

    x = layers.Dropout(0.20, seed=seed + 2, name="dropout_base")(x)

    out = layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 3),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="residual_output",
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="gru_residual_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        weighted_metrics=[],
    )

    return model


# =========================================================
# MÉTRICAS
# =========================================================

def masked_vector(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    mask = np.asarray(w, dtype=float) > 0
    out = arr[mask]
    return out[np.isfinite(out)]


def sd_curve(residual_matrix: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    sd = np.full(residual_matrix.shape[1], np.nan, dtype=float)

    for j in range(residual_matrix.shape[1]):
        values = masked_vector(residual_matrix[:, j], weight_matrix[:, j])
        if values.size >= 2:
            sd[j] = np.std(values, ddof=1)

    return sd


def metric_curve(residual_matrix: np.ndarray, weight_matrix: np.ndarray, metric: str) -> np.ndarray:
    metric = metric.upper()
    out = np.full(residual_matrix.shape[1], np.nan, dtype=float)

    for j in range(residual_matrix.shape[1]):
        values = masked_vector(residual_matrix[:, j], weight_matrix[:, j])
        if values.size < 2:
            continue

        if metric == "MAE":
            out[j] = np.mean(np.abs(values))
        elif metric == "MSE":
            out[j] = np.mean(values ** 2)
        elif metric == "RMSE":
            out[j] = np.sqrt(np.mean(values ** 2))
        elif metric == "SD":
            out[j] = np.std(values, ddof=1)
        else:
            raise ValueError(f"Métrica no reconocida: {metric}")

    return out


def r2_curve(y_true: np.ndarray, y_pred: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[1], np.nan, dtype=float)

    for j in range(y_true.shape[1]):
        mask = weight_matrix[:, j] > 0
        yt = y_true[mask, j]
        yp = y_pred[mask, j]
        valid = np.isfinite(yt) & np.isfinite(yp)

        if valid.sum() >= 3:
            try:
                out[j] = r2_score(yt[valid], yp[valid])
            except Exception:
                out[j] = np.nan

    return out


def summarize_metrics(periods: np.ndarray, y_true_resid: np.ndarray, yhat_resid: np.ndarray, W: np.ndarray) -> pd.DataFrame:
    corrected_resid = np.where(W > 0, y_true_resid - yhat_resid, np.nan)

    data = {
        "Period": periods,
        "SD_base": metric_curve(y_true_resid, W, "SD"),
        "SD_model": metric_curve(corrected_resid, W, "SD"),
        "MAE_base": metric_curve(y_true_resid, W, "MAE"),
        "MAE_model": metric_curve(corrected_resid, W, "MAE"),
        "MSE_base": metric_curve(y_true_resid, W, "MSE"),
        "MSE_model": metric_curve(corrected_resid, W, "MSE"),
        "RMSE_base": metric_curve(y_true_resid, W, "RMSE"),
        "RMSE_model": metric_curve(corrected_resid, W, "RMSE"),
        "R2_residual_prediction": r2_curve(y_true_resid, yhat_resid, W),
    }

    df = pd.DataFrame(data)
    df["SD_reduction_pct"] = np.where(
        df["SD_base"] > 0,
        100.0 * (df["SD_base"] - df["SD_model"]) / df["SD_base"],
        np.nan,
    )

    return df


# =========================================================
# ENTRENAMIENTO
# =========================================================

def train_full_model(
    df_wide: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    periods: np.ndarray,
    model_key: str,
    epochs: int,
    batch_size: int = 64,
    verbose: int = 0,
    seed: int = SEED,
) -> Dict[str, Any]:
    reset_reproducibility(seed)
    tf.keras.backend.clear_session()

    X_df = get_feature_matrix(df_wide, model_key)

    pre = build_preprocessor(model_key)
    Xp = pre.fit_transform(X_df)
    X_seq = build_sequence_input(np.asarray(Xp, dtype=np.float32), periods)

    model = build_residual_model(seq_len=len(periods), feat_dim=X_seq.shape[-1], seed=seed)

    hist = model.fit(
        X_seq,
        Y[..., None],
        sample_weight=W,
        epochs=int(epochs),
        batch_size=batch_size,
        verbose=verbose,
        shuffle=False,
    )

    Yhat = model.predict(X_seq, verbose=0).squeeze(-1)
    metrics = summarize_metrics(periods, Y, Yhat, W)

    return {
        "model": model,
        "preprocessor": pre,
        "history": hist.history,
        "Yhat_full": Yhat,
        "metrics_full": metrics,
    }


def run_logo_experiment_fixed_epochs(
    df_wide: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    periods: np.ndarray,
    model_key: str,
    epochs: int,
    batch_size: int = 64,
    verbose: int = 0,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    LOGO por evento, equivalente al workflow de notebooks:
    para cada evento se entrena con los demás y se predice OOF.
    """
    logo = LeaveOneGroupOut()
    groups = df_wide[EVENT_COL].to_numpy()

    Yhat_oof = np.full_like(Y, np.nan, dtype=np.float32)

    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(logo.split(df_wide, Y, groups=groups), start=1):
        reset_reproducibility(seed + fold)
        tf.keras.backend.clear_session()

        df_tr = df_wide.iloc[tr_idx].reset_index(drop=True)
        df_va = df_wide.iloc[va_idx].reset_index(drop=True)

        X_tr = get_feature_matrix(df_tr, model_key)
        X_va = get_feature_matrix(df_va, model_key)

        pre = build_preprocessor(model_key)
        Xp_tr = pre.fit_transform(X_tr)
        Xp_va = pre.transform(X_va)

        Xseq_tr = build_sequence_input(np.asarray(Xp_tr, dtype=np.float32), periods)
        Xseq_va = build_sequence_input(np.asarray(Xp_va, dtype=np.float32), periods)

        model = build_residual_model(seq_len=len(periods), feat_dim=Xseq_tr.shape[-1], seed=seed + fold)

        model.fit(
            Xseq_tr,
            Y[tr_idx][..., None],
            sample_weight=W[tr_idx],
            epochs=int(epochs),
            batch_size=batch_size,
            verbose=verbose,
            shuffle=False,
        )

        pred_va = model.predict(Xseq_va, verbose=0).squeeze(-1)
        Yhat_oof[va_idx] = pred_va

        eps_va = np.where(W[va_idx] > 0, Y[va_idx] - pred_va, np.nan)

        fold_rows.append(
            {
                "fold": fold,
                "event": str(df_wide.iloc[va_idx][EVENT_COL].iloc[0]),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "valid_sd_mean": float(np.nanmean(sd_curve(eps_va, W[va_idx]))),
                "valid_mae_mean": float(np.nanmean(metric_curve(eps_va, W[va_idx], "MAE"))),
                "valid_rmse_mean": float(np.nanmean(metric_curve(eps_va, W[va_idx], "RMSE"))),
            }
        )

        print(f"[{model_key}] Fold {fold:02d} | event={fold_rows[-1]['event']} | n_valid={len(va_idx)}")

    metrics_oof = summarize_metrics(periods, Y, Yhat_oof, W)

    return {
        "Yhat_oof": Yhat_oof,
        "metrics_oof": metrics_oof,
        "fold_summary": pd.DataFrame(fold_rows),
    }


# =========================================================
# EMPAQUETADO DEL MODELO
# =========================================================

def pack_keras_model(model: tf.keras.Model) -> Dict[str, Any]:
    return {
        "model_json": model.to_json(),
        "weights": model.get_weights(),
    }


def unpack_keras_model(model_pack: Dict[str, Any]) -> tf.keras.Model:
    model = tf.keras.models.model_from_json(model_pack["model_json"])
    model.set_weights(model_pack["weights"])
    return model


def save_model_package(package: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(package, f)


def load_model_package(model_path: str | Path = "model/model.pkl") -> Dict[str, Any]:
    path = resolve_path(model_path)
    with open(path, "rb") as f:
        return pickle.load(f)


def build_single_model_artifact(
    model_key: str,
    data_path: str | Path,
    meta_path: Optional[str | Path],
    epochs: int,
    batch_size: int,
    run_logo: bool,
    verbose: int,
    seed: int,
) -> Dict[str, Any]:
    print(f"\nEntrenando modelo: {model_key}")

    df_long = load_long_dataset(data_path=data_path, meta_path=meta_path, model_key=model_key)
    df_wide, periods, Y_raw, W, Y, Sa_raw = build_wide_dataset(df_long, model_key)

    logo_artifact = None
    if run_logo:
        logo_artifact = run_logo_experiment_fixed_epochs(
            df_wide=df_wide,
            Y=Y,
            W=W,
            periods=periods,
            model_key=model_key,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            seed=seed,
        )

    full_artifact = train_full_model(
        df_wide=df_wide,
        Y=Y,
        W=W,
        periods=periods,
        model_key=model_key,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        seed=seed,
    )

    cfg = MODEL_CONFIGS[model_key]
    feature_cols = cfg["continuous_cols"] + cfg["cat_cols"]

    artifact = {
        "model_key": model_key,
        "label": cfg["label"],
        "feature_cols": feature_cols,
        "continuous_cols": cfg["continuous_cols"],
        "cat_cols": cfg["cat_cols"],
        "periods": periods,
        "model": pack_keras_model(full_artifact["model"]),
        "preprocessor": full_artifact["preprocessor"],
        "history": full_artifact["history"],
        "metrics_full": full_artifact["metrics_full"],
        "metrics_oof": logo_artifact["metrics_oof"] if logo_artifact else None,
        "fold_summary": logo_artifact["fold_summary"] if logo_artifact else None,
        "Yhat_oof": logo_artifact["Yhat_oof"] if logo_artifact else None,
        "Yhat_full": full_artifact["Yhat_full"],
        "Y": Y,
        "W": W,
        "Sa_raw": Sa_raw,
        "df_wide_preview": df_wide.head(200).copy(),
        "dataset_summary": {
            "n_long_rows": int(len(df_long)),
            "n_records": int(len(df_wide)),
            "n_events": int(df_wide[EVENT_COL].nunique()),
            "n_stations": int(df_wide[STATION_COL].nunique()),
            "n_periods": int(len(periods)),
            "valid_periods_mean": float(W.sum(axis=1).mean()),
        },
    }

    return artifact


def train_all(
    data_path: str | Path,
    meta_path: Optional[str | Path],
    output_path: str | Path,
    models: Iterable[str],
    epochs: int,
    batch_size: int,
    run_logo: bool,
    verbose: int,
    seed: int,
) -> Dict[str, Any]:
    reset_reproducibility(seed)

    package = {
        "created_by": "train_model.py",
        "seed": seed,
        "models": {},
        "model_configs": MODEL_CONFIGS,
    }

    for model_key in models:
        model_key = model_key.strip()
        if not model_key:
            continue

        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Modelo no reconocido: {model_key}. Opciones: {list(MODEL_CONFIGS)}")

        try:
            artifact = build_single_model_artifact(
                model_key=model_key,
                data_path=data_path,
                meta_path=meta_path,
                epochs=epochs,
                batch_size=batch_size,
                run_logo=run_logo,
                verbose=verbose,
                seed=seed,
            )
            package["models"][model_key] = artifact
        except ImportError as exc:
            print(f"[ADVERTENCIA] No se entrenó {model_key}: {exc}")
        except Exception as exc:
            print(f"[ERROR] Falló el entrenamiento de {model_key}: {exc}")
            raise

    if not package["models"]:
        raise RuntimeError("No se entrenó ningún modelo. Revisa dependencias, rutas y columnas.")

    save_model_package(package, output_path)
    print(f"\nModelo guardado en: {output_path}")

    return package


# =========================================================
# PREDICCIÓN
# =========================================================

def build_prediction_frame(
    model_key: str,
    magnitude: float,
    rrup_km: float,
    zhypo_km: float,
    soil_class: int,
    rvolc_km: float = 0.0,
    station_elevation_m: float = 1.0,
) -> pd.DataFrame:
    cfg = MODEL_CONFIGS[model_key]

    row = {
        "Hypocenter Depth (km)": float(zhypo_km),
        "Magnitude": float(magnitude),
        "Rrup_OpenQuake": np.log(float(rrup_km)),
        SOIL_COL: int(soil_class),
    }

    if RVOLC_COL in cfg["continuous_cols"]:
        row[RVOLC_COL] = np.log1p(float(rvolc_km))

    if LOG_ELEV_COL in cfg["continuous_cols"]:
        row[LOG_ELEV_COL] = np.log(max(float(station_elevation_m), 1e-6))

    return pd.DataFrame([row])


def predict_residual_spectrum(
    package: Dict[str, Any],
    model_key: str,
    magnitude: float,
    rrup_km: float,
    zhypo_km: float,
    soil_class: int,
    rvolc_km: float = 0.0,
    station_elevation_m: float = 1.0,
) -> Dict[str, Any]:
    if model_key not in package["models"]:
        raise ValueError(f"El paquete no contiene el modelo {model_key}. Modelos disponibles: {list(package['models'])}")

    artifact = package["models"][model_key]
    model = unpack_keras_model(artifact["model"])
    pre = artifact["preprocessor"]
    periods = np.asarray(artifact["periods"], dtype=float)

    x_df = build_prediction_frame(
        model_key=model_key,
        magnitude=magnitude,
        rrup_km=rrup_km,
        zhypo_km=zhypo_km,
        soil_class=soil_class,
        rvolc_km=rvolc_km,
        station_elevation_m=station_elevation_m,
    )

    Xp = pre.transform(x_df[artifact["feature_cols"]])
    X_seq = build_sequence_input(np.asarray(Xp, dtype=np.float32), periods)

    residual_hat = model.predict(X_seq, verbose=0).squeeze()

    base_sa = None
    corrected_sa = None
    base_name = None

    if model_key in ["nosam", "nosam_elevation"]:
        try:
            from NoSAm_GMMs_2024 import NoSAm_Crustal_2023

            base_sa = np.array(
                [
                    NoSAm_Crustal_2023(
                        float(t),
                        float(magnitude),
                        float(rrup_km),
                        int(soil_class),
                        "average",
                        float(zhypo_km),
                        float(rvolc_km),
                    )[0]
                    for t in periods
                ],
                dtype=float,
            )
            corrected_sa = base_sa * np.exp(residual_hat)
            base_name = "NoSAm"
        except Exception:
            base_sa = None
            corrected_sa = None
            base_name = "NoSAm no disponible"

    elif model_key == "ask14":
        try:
            vs30 = soil_class_to_vs30(int(soil_class))
            ln_base = ask14_ln_spectrum(
                periods=periods,
                Mag=float(magnitude),
                Rrup=float(rrup_km),
                Zhypo=float(zhypo_km),
                Vs30_Val=vs30,
            )
            base_sa = np.exp(ln_base)
            corrected_sa = base_sa * np.exp(residual_hat)
            base_name = "ASK14"
        except Exception:
            base_sa = None
            corrected_sa = None
            base_name = "ASK14 no disponible"

    return {
        "model_key": model_key,
        "label": artifact["label"],
        "periods": periods,
        "residual_hat": residual_hat,
        "base_sa": base_sa,
        "corrected_sa": corrected_sa,
        "base_name": base_name,
    }


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelos GRU residuales para espectros de respuesta.")

    parser.add_argument("--data", default="Resids_for_Eliasib.xlsx", help="Ruta de la base larga principal.")
    parser.add_argument("--meta", default="CopiaDataBaseSGC2.xlsx", help="Ruta de la base de metadatos/elevación.")
    parser.add_argument("--output", default="model/model.pkl", help="Ruta de salida del paquete entrenado.")
    parser.add_argument(
        "--models",
        default="nosam,nosam_elevation",
        help="Modelos a entrenar separados por coma: nosam,nosam_elevation,ask14.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Épocas fijas para LOGO y full train.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--skip-logo", action="store_true", help="Omite LOGO para pruebas rápidas.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose de Keras.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla reproducible.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    train_all(
        data_path=args.data,
        meta_path=args.meta,
        output_path=args.output,
        models=model_keys,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_logo=not args.skip_logo,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```


## `tabs/introduccion.py`

```python
from dash import html
import dash_bootstrap_components as dbc


def metric_card(title, text, icon):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(className=f"fa-solid {icon} card-icon"),
                html.H4(title, className="card-title"),
                html.P(text, className="card-text"),
            ]
        ),
        className="soft-card h-100",
    )


def layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Ingeniería sísmica + ciencia de datos", className="section-kicker"),
                            html.H1("Predicción de aceleraciones espectrales mediante aprendizaje residual", className="page-title"),
                            html.P(
                                "Este dashboard resume un flujo de modelación para espectros de respuesta RotD50. "
                                "La idea central es partir de un modelo físico o empírico base, como NoSAm o ASK14, "
                                "y entrenar una red GRU para aprender el residuo logarítmico entre el espectro observado "
                                "y el espectro estimado por la GMPE.",
                                className="lead",
                            ),
                        ],
                        lg=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Arquitectura del enfoque", className="mb-3"),
                                    html.Ul(
                                        [
                                            html.Li("Entrada: magnitud, distancia Rrup, profundidad hipocentral, clase de suelo y variables regionales."),
                                            html.Li("Salida: secuencia de residuos o espectro completo en 22 períodos."),
                                            html.Li("Máscara: Tmax limita los períodos confiables por registro."),
                                            html.Li("Validación: Leave-One-Group-Out por evento sísmico."),
                                        ],
                                        className="mb-0",
                                    ),
                                ]
                            ),
                            className="highlight-card",
                        ),
                        lg=4,
                    ),
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(metric_card("Espectro de respuesta", "Representa la demanda máxima esperada de un oscilador SDOF para diferentes períodos.", "fa-wave-square"), md=4),
                    dbc.Col(metric_card("Residual learning", "La GRU no reemplaza la GMPE: aprende la corrección residual en escala logarítmica.", "fa-brain"), md=4),
                    dbc.Col(metric_card("Máscara Tmax", "Ignora períodos no válidos o poco confiables durante entrenamiento y evaluación.", "fa-filter"), md=4),
                ],
                className="g-4",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Interpretación del target", className="mb-3"),
                                html.P(
                                    "Para los modelos residuales, el objetivo principal es el residuo logarítmico: "
                                    "residuo = ln(Sa observado) - ln(Sa GMPE). "
                                    "Luego, para predecir el espectro corregido, se combina la GMPE con la predicción residual: "
                                    "ln(Sa corregido) = ln(Sa GMPE) + residuo_predicho.",
                                    className="mb-0",
                                ),
                            ]
                        ),
                        className="soft-card mt-4",
                    ),
                    width=12,
                )
            ),
        ],
        fluid=True,
    )
```


## `tabs/problema.py`

```python
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from model.train_model import PERIOD_COL, SA_COL, TMAX_COL, standardize_columns, read_table


def load_problem_data():
    try:
        df = read_table("Resids_for_Eliasib.xlsx")
        return standardize_columns(df)
    except Exception as exc:
        return exc


def layout():
    data = load_problem_data()

    if isinstance(data, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("No se pudo cargar la base principal.", className="alert-heading"),
                    html.P(str(data)),
                    html.P("Coloca Resids_for_Eliasib.xlsx en la raíz del proyecto o en la carpeta data/."),
                ],
                color="warning",
            ),
            fluid=True,
        )

    df = data.copy()
    df["ln_Sa"] = np.log(df[SA_COL].astype(float).clip(lower=1e-12))

    fig_hist = px.histogram(
        df,
        x="ln_Sa",
        nbins=60,
        title="Distribución de ln(Sa) observado",
        labels={"ln_Sa": "ln(Sa)", "count": "Frecuencia"},
    )
    fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    valid_by_period = (
        df.assign(is_valid=df[PERIOD_COL] <= df[TMAX_COL])
        .groupby(PERIOD_COL, as_index=False)["is_valid"]
        .mean()
        .rename(columns={"is_valid": "fraction_valid"})
    )

    fig_mask = px.line(
        valid_by_period,
        x=PERIOD_COL,
        y="fraction_valid",
        markers=True,
        title="Fracción de valores válidos por período usando Tmax",
        labels={PERIOD_COL: "Período T (s)", "fraction_valid": "Fracción válida"},
    )
    fig_mask.update_xaxes(type="log")
    fig_mask.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))

    summary_cards = dbc.Row(
        [
            dbc.Col(dbc.Card(dbc.CardBody([html.H3(f"{df['Record Sequence Number'].nunique():,}"), html.P("registros")] ), className="metric-card"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.H3(f"{df['EQID_Code'].nunique():,}"), html.P("eventos")] ), className="metric-card"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.H3(f"{df[PERIOD_COL].nunique():,}"), html.P("períodos")] ), className="metric-card"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.H3(f"{df['Station Code'].nunique():,}"), html.P("estaciones")] ), className="metric-card"), md=3),
        ],
        className="g-4 mb-4",
    )

    return dbc.Container(
        [
            html.Div("Diagnóstico de la variable objetivo", className="section-kicker"),
            html.H1("Comportamiento del espectro observado y de la máscara Tmax", className="page-title"),
            html.P(
                "La variable Sa suele presentar fuerte asimetría positiva en escala física. "
                "Por eso el modelado residual trabaja en escala logarítmica y respeta la disponibilidad por período.",
                className="lead",
            ),
            summary_cards,
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_hist)), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_mask)), className="soft-card"), lg=6),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )
```


## `tabs/objetivos.py`

```python
from dash import html
import dash_bootstrap_components as dbc


def layout():
    objectives = [
        ("Objetivo general", "Desarrollar una aplicación analítica que integre EDA, validación y predicción de aceleraciones espectrales mediante modelos residuales basados en GRU."),
        ("Objetivo específico 1", "Preparar el dataset largo, transformar variables predictoras y construir matrices espectrales respetando la máscara Tmax."),
        ("Objetivo específico 2", "Entrenar modelos residuales comparables bajo validación LOGO por evento para evaluar generalización sísmica."),
        ("Objetivo específico 3", "Visualizar métricas por período: SD, RMSE, MAE, MSE, R², reales vs predichos y distribución de residuos."),
        ("Objetivo específico 4", "Permitir predicción interactiva para escenarios definidos por magnitud, distancia, profundidad, clase de suelo y variables regionales."),
    ]

    return dbc.Container(
        [
            html.Div("Propósito metodológico", className="section-kicker"),
            html.H1("Objetivos y justificación", className="page-title"),
            html.P(
                "La aplicación organiza el flujo de investigación en una herramienta reproducible: carga de datos, "
                "entrenamiento, validación cruzada por evento y predicción interactiva.",
                className="lead",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(title, className="mb-3"),
                                    html.P(text, className="mb-0"),
                                ]
                            ),
                            className="soft-card h-100",
                        ),
                        md=6,
                    )
                    for title, text in objectives
                ],
                className="g-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Justificación", className="mb-3"),
                        html.P(
                            "El aprendizaje residual es útil cuando se desea preservar la estructura de una GMPE y, al mismo tiempo, "
                            "capturar patrones regionales o sistemáticos que no quedan plenamente representados por el modelo base. "
                            "La validación LOGO por evento evita que registros del mismo sismo aparezcan simultáneamente en entrenamiento "
                            "y validación, por lo que es más exigente que una partición aleatoria por registro.",
                            className="mb-0",
                        ),
                    ]
                ),
                className="highlight-card mt-4",
            ),
        ],
        fluid=True,
    )
```


## `tabs/resultados.py`

```python
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from model.train_model import load_model_package


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False)
    return fig


def get_package():
    try:
        return load_model_package("model/model.pkl")
    except Exception as exc:
        return exc


def model_options(package):
    if isinstance(package, Exception):
        return []
    return [{"label": artifact["label"], "value": key} for key, artifact in package["models"].items()]


def metric_figure(metrics, metric_name):
    if metrics is None or len(metrics) == 0:
        return empty_figure("No hay métricas LOGO. Entrena con LOGO o revisa model.pkl.")

    df = metrics.copy()
    fig = go.Figure()

    base_col = f"{metric_name}_base"
    model_col = f"{metric_name}_model"

    if base_col in df.columns:
        fig.add_trace(go.Scatter(x=df["Period"], y=df[base_col], mode="lines+markers", name=f"{metric_name} base"))
    if model_col in df.columns:
        fig.add_trace(go.Scatter(x=df["Period"], y=df[model_col], mode="lines+markers", name=f"{metric_name} corregido"))

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(title=metric_name)
    fig.update_layout(template="plotly_white", title=f"{metric_name} por período", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def sd_reduction_figure(metrics):
    if metrics is None or "SD_reduction_pct" not in metrics.columns:
        return empty_figure("No hay reducción de SD disponible.")

    fig = px.line(
        metrics,
        x="Period",
        y="SD_reduction_pct",
        markers=True,
        title="Reducción porcentual de SD por período",
        labels={"Period": "Período T (s)", "SD_reduction_pct": "Reducción SD (%)"},
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_xaxes(type="log")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def residual_scatter_figure(artifact):
    Y = np.asarray(artifact["Y"], dtype=float)
    W = np.asarray(artifact["W"], dtype=float)
    Yhat = artifact.get("Yhat_oof")
    title_suffix = "OOF"

    if Yhat is None:
        Yhat = artifact.get("Yhat_full")
        title_suffix = "full train"

    Yhat = np.asarray(Yhat, dtype=float)

    j = 0
    mask = W[:, j] > 0
    df = pd.DataFrame(
        {
            "Real residual": Y[mask, j],
            "Predicted residual": Yhat[mask, j],
        }
    ).dropna()

    if df.empty:
        return empty_figure("No hay predicciones para graficar.")

    fig = px.scatter(
        df,
        x="Real residual",
        y="Predicted residual",
        trendline="ols",
        title=f"Residuos reales vs predichos en T inicial ({title_suffix})",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def residual_hist_figure(artifact):
    Y = np.asarray(artifact["Y"], dtype=float)
    W = np.asarray(artifact["W"], dtype=float)
    Yhat = artifact.get("Yhat_oof")
    title_suffix = "OOF"

    if Yhat is None:
        Yhat = artifact.get("Yhat_full")
        title_suffix = "full train"

    eps_corr = np.where(W > 0, Y - np.asarray(Yhat, dtype=float), np.nan)
    values = eps_corr[np.isfinite(eps_corr)]

    if values.size == 0:
        return empty_figure("No hay residuos corregidos para graficar.")

    fig = px.histogram(
        x=values,
        nbins=70,
        title=f"Distribución global de residuos corregidos ({title_suffix})",
        labels={"x": "Residuo corregido"},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def layout():
    package = get_package()

    if isinstance(package, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("Aún no hay model/model.pkl disponible.", className="alert-heading"),
                    html.P(str(package)),
                    html.P("Ejecuta primero: python model/train_model.py --models nosam,nosam_elevation --epochs 60"),
                ],
                color="warning",
            ),
            fluid=True,
        )

    opts = model_options(package)
    default_model = opts[0]["value"] if opts else None

    return dbc.Container(
        [
            html.Div("Evaluación del modelo", className="section-kicker"),
            html.H1("Resultados: EDA, validación LOGO y métricas por período", className="page-title"),
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
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-sd")), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-rmse")), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-mae")), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-sd-reduction")), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-scatter")), className="soft-card"), lg=6),
                    dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fig-residual-hist")), className="soft-card"), lg=6),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )


def register_callbacks(app):
    @app.callback(
        Output("fig-sd", "figure"),
        Output("fig-rmse", "figure"),
        Output("fig-mae", "figure"),
        Output("fig-sd-reduction", "figure"),
        Output("fig-scatter", "figure"),
        Output("fig-residual-hist", "figure"),
        Input("results-model-dropdown", "value"),
    )
    def update_results(model_key):
        package = get_package()
        if isinstance(package, Exception) or model_key not in package["models"]:
            fig = empty_figure("Modelo no disponible.")
            return fig, fig, fig, fig, fig, fig

        artifact = package["models"][model_key]
        metrics = artifact.get("metrics_oof")
        if metrics is None:
            metrics = artifact.get("metrics_full")

        return (
            metric_figure(metrics, "SD"),
            metric_figure(metrics, "RMSE"),
            metric_figure(metrics, "MAE"),
            sd_reduction_figure(metrics),
            residual_scatter_figure(artifact),
            residual_hist_figure(artifact),
        )
```


## `tabs/prediccion.py`

```python
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

from model.train_model import load_model_package, predict_residual_spectrum


def get_package():
    try:
        return load_model_package("model/model.pkl")
    except Exception as exc:
        return exc


def model_options(package):
    if isinstance(package, Exception):
        return []
    return [{"label": artifact["label"], "value": key} for key, artifact in package["models"].items()]


def prediction_figure(result):
    periods = np.asarray(result["periods"], dtype=float)
    residual_hat = np.asarray(result["residual_hat"], dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=residual_hat, mode="lines+markers", name="Residuo predicho"))

    if result.get("corrected_sa") is not None:
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=np.asarray(result["corrected_sa"], dtype=float),
                mode="lines+markers",
                name="Sa corregido",
                yaxis="y2",
            )
        )

        fig.update_layout(
            yaxis2=dict(title="Sa corregido", overlaying="y", side="right", type="log"),
        )

    fig.update_xaxes(type="log", title="Período T (s)")
    fig.update_yaxes(title="Residuo logarítmico predicho")
    fig.update_layout(template="plotly_white", title="Predicción espectral", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def layout():
    package = get_package()

    if isinstance(package, Exception):
        return dbc.Container(
            dbc.Alert(
                [
                    html.H5("No hay modelo entrenado para predicción.", className="alert-heading"),
                    html.P(str(package)),
                    html.P("Ejecuta primero: python model/train_model.py --models nosam,nosam_elevation --epochs 60"),
                ],
                color="warning",
            ),
            fluid=True,
        )

    opts = model_options(package)
    default_model = opts[0]["value"] if opts else None

    return dbc.Container(
        [
            html.Div("Predicción interactiva", className="section-kicker"),
            html.H1("Predicción en tiempo real del primer período y espectro residual", className="page-title"),
            html.P(
                "Selecciona el modelo y define un escenario sísmico. La red predice el residuo logarítmico en los 22 períodos. "
                "Si la GMPE base está disponible en el entorno local, también se calcula Sa corregido.",
                className="lead",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Modelo"),
                                    dcc.Dropdown(id="pred-model", options=opts, value=default_model, clearable=False),
                                    html.Hr(),
                                    html.Label("Magnitud Mw"),
                                    dbc.Input(id="pred-magnitude", type="number", value=5.5, min=3.0, max=8.5, step=0.1),
                                    html.Label("Rrup (km)", className="mt-3"),
                                    dbc.Input(id="pred-rrup", type="number", value=100.0, min=0.1, step=1.0),
                                    html.Label("Profundidad hipocentral (km)", className="mt-3"),
                                    dbc.Input(id="pred-zhypo", type="number", value=15.0, min=0.0, step=1.0),
                                    html.Label("Clase de suelo", className="mt-3"),
                                    dcc.Dropdown(
                                        id="pred-soil",
                                        options=[
                                            {"label": "1", "value": 1},
                                            {"label": "2", "value": 2},
                                            {"label": "3", "value": 3},
                                            {"label": "4", "value": 4},
                                            {"label": "5", "value": 5},
                                        ],
                                        value=3,
                                        clearable=False,
                                    ),
                                    html.Label("Rvolc (km)", className="mt-3"),
                                    dbc.Input(id="pred-rvolc", type="number", value=0.0, min=0.0, step=1.0),
                                    html.Label("Station Elevation (m)", className="mt-3"),
                                    dbc.Input(id="pred-elev", type="number", value=100.0, min=0.0, step=10.0),
                                    dbc.Button("Predecir", id="pred-button", color="primary", className="w-100 mt-4"),
                                ]
                            ),
                            className="soft-card",
                        ),
                        lg=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(dbc.CardBody(html.Div(id="prediction-summary")), className="highlight-card mb-4"),
                            dbc.Card(dbc.CardBody(dcc.Graph(id="prediction-graph")), className="soft-card"),
                        ],
                        lg=8,
                    ),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )


def register_callbacks(app):
    @app.callback(
        Output("prediction-summary", "children"),
        Output("prediction-graph", "figure"),
        Input("pred-button", "n_clicks"),
        State("pred-model", "value"),
        State("pred-magnitude", "value"),
        State("pred-rrup", "value"),
        State("pred-zhypo", "value"),
        State("pred-soil", "value"),
        State("pred-rvolc", "value"),
        State("pred-elev", "value"),
        prevent_initial_call=False,
    )
    def update_prediction(n_clicks, model_key, magnitude, rrup, zhypo, soil, rvolc, elev):
        package = get_package()
        if isinstance(package, Exception):
            fig = go.Figure()
            fig.add_annotation(text="Modelo no disponible.", x=0.5, y=0.5, showarrow=False)
            return dbc.Alert(str(package), color="warning"), fig

        try:
            result = predict_residual_spectrum(
                package=package,
                model_key=model_key,
                magnitude=float(magnitude),
                rrup_km=float(rrup),
                zhypo_km=float(zhypo),
                soil_class=int(soil),
                rvolc_km=float(rvolc or 0.0),
                station_elevation_m=float(elev or 1.0),
            )

            first_period = float(result["periods"][0])
            first_residual = float(result["residual_hat"][0])

            rows = [
                html.H4(result["label"], className="mb-3"),
                html.P(f"Primer período: T = {first_period:.3f} s"),
                html.P(f"Residuo predicho en el primer período: {first_residual:.4f}"),
            ]

            if result.get("corrected_sa") is not None:
                rows.append(html.P(f"GMPE base: {result['base_name']}"))
                rows.append(html.P(f"Sa corregido en el primer período: {float(result['corrected_sa'][0]):.6g}"))
            else:
                rows.append(
                    dbc.Alert(
                        "Se calculó el residuo, pero no Sa corregido porque la GMPE base no está disponible en este entorno.",
                        color="info",
                        className="mt-2",
                    )
                )

            return rows, prediction_figure(result)

        except Exception as exc:
            fig = go.Figure()
            fig.add_annotation(text=str(exc), x=0.5, y=0.5, showarrow=False)
            fig.update_layout(template="plotly_white")
            return dbc.Alert(str(exc), color="danger"), fig
```


## `tabs/limitaciones.py`

```python
from dash import html
import dash_bootstrap_components as dbc


def layout():
    limitations = [
        ("Tmax y disponibilidad espectral", "Los períodos mayores que Tmax se excluyen del entrenamiento y de la evaluación. Esto reduce información efectiva en períodos largos."),
        ("Generalización por evento", "LOGO por evento es exigente: algunos eventos tienen pocos registros, lo que puede aumentar la variabilidad de las métricas fold a fold."),
        ("Dependencia de la GMPE base", "El modelo residual corrige una GMPE; por tanto, la calidad de la predicción final también depende de la consistencia del modelo base."),
        ("ASK14 requiere OpenQuake", "El entrenamiento o predicción de ASK14 necesita instalar OpenQuake. Si no está disponible, el dashboard puede operar con los modelos NoSAm."),
        ("Extrapolación", "Predicciones fuera del dominio de magnitud, distancia, profundidad o clase de suelo observados deben interpretarse con cautela."),
    ]

    return dbc.Container(
        [
            html.Div("Alcance del sistema", className="section-kicker"),
            html.H1("Limitaciones", className="page-title"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(title),
                                    html.P(text, className="mb-0"),
                                ]
                            ),
                            className="soft-card h-100",
                        ),
                        md=6,
                    )
                    for title, text in limitations
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )
```


## `tabs/conclusiones.py`

```python
from dash import html
import dash_bootstrap_components as dbc


def layout():
    return dbc.Container(
        [
            html.Div("Cierre técnico", className="section-kicker"),
            html.H1("Conclusiones", className="page-title"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            "El dashboard consolida un flujo reproducible para analizar y predecir aceleraciones espectrales "
                            "a partir de modelos residuales. La arquitectura modular permite mantener cada pestaña en un archivo "
                            "independiente, facilitando mantenimiento, expansión y despliegue.",
                        ),
                        html.P(
                            "El uso de GRU es coherente con la naturaleza secuencial del espectro: cada registro se representa "
                            "como una secuencia ordenada de períodos, y la red aprende una corrección residual dependiente del período.",
                        ),
                        html.P(
                            "La validación LOGO por evento permite evaluar mejor la capacidad de generalización ante sismos no vistos. "
                            "Las métricas por período ayudan a identificar en qué rangos espectrales la corrección residual aporta mayor reducción de error.",
                            className="mb-0",
                        ),
                    ]
                ),
                className="highlight-card",
            ),
        ],
        fluid=True,
    )
```


## `assets/styles.css`

```css
:root {
  --pastel-blue: #d7ebff;
  --pastel-purple: #eadcff;
  --pastel-green: #dbf7ea;
  --ink: #243447;
  --muted: #6c7a89;
  --surface: #ffffff;
  --soft-border: #e8eef5;
}

body {
  background: linear-gradient(135deg, #f7fbff 0%, #fbf7ff 50%, #f8fffb 100%);
  color: var(--ink);
  font-family: "Inter", "Segoe UI", Arial, sans-serif;
}

.top-navbar {
  background: rgba(255, 255, 255, 0.88);
  border-bottom: 1px solid var(--soft-border);
  backdrop-filter: blur(8px);
  padding: 0.85rem 0;
}

.main-container {
  padding: 1.5rem 2rem 3rem 2rem;
}

.brand-badge {
  width: 48px;
  height: 48px;
  border-radius: 16px;
  background: linear-gradient(135deg, var(--pastel-blue), var(--pastel-purple));
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  color: #31506f;
  box-shadow: 0 8px 24px rgba(49, 80, 111, 0.12);
}

.brand-title {
  font-size: 1.15rem;
  font-weight: 800;
}

.brand-subtitle {
  font-size: 0.85rem;
  color: var(--muted);
}

.tabs-card,
.soft-card,
.highlight-card,
.metric-card {
  border: 1px solid var(--soft-border);
  border-radius: 24px;
  box-shadow: 0 12px 32px rgba(48, 72, 102, 0.08);
}

.tabs-card {
  background: rgba(255, 255, 255, 0.84);
  margin-bottom: 1.25rem;
}

.custom-tabs {
  border: none !important;
}

.custom-tab {
  border: none !important;
  padding: 0.85rem 1rem !important;
  color: var(--muted) !important;
  background: transparent !important;
  font-weight: 650;
}

.custom-tab-selected {
  color: #255f91 !important;
  background: var(--pastel-blue) !important;
  border-radius: 16px !important;
}

.tab-content-wrapper {
  margin-top: 1rem;
}

.section-kicker {
  color: #6879d8;
  text-transform: uppercase;
  font-weight: 800;
  letter-spacing: 0.08em;
  font-size: 0.78rem;
  margin-bottom: 0.35rem;
}

.page-title {
  font-weight: 850;
  letter-spacing: -0.035em;
  margin-bottom: 0.75rem;
}

.lead {
  color: var(--muted);
  max-width: 980px;
}

.soft-card {
  background: rgba(255, 255, 255, 0.94);
}

.highlight-card {
  background: linear-gradient(135deg, var(--pastel-blue), var(--pastel-green));
}

.metric-card {
  text-align: center;
  background: #fff;
}

.metric-card h3 {
  font-weight: 850;
  color: #255f91;
}

.card-icon {
  width: 48px;
  height: 48px;
  background: var(--pastel-blue);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #31506f;
  margin-bottom: 1rem;
  font-size: 1.15rem;
}

.card-title {
  font-weight: 800;
}

.card-text {
  color: var(--muted);
}

label {
  font-weight: 700;
  color: #37485c;
  margin-bottom: 0.35rem;
}
```


## `requirements.txt`

```text
dash>=2.16
dash-bootstrap-components>=1.5
plotly>=5.20
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
tensorflow>=2.15
openpyxl>=3.1
scipy>=1.10
statsmodels>=0.14
joblib>=1.3

# Opcional, necesario solo para el modelo ASK14:
# openquake.engine
```


## `README.md`

```markdown
# Dashboard de aceleraciones espectrales

Proyecto Dash modular para análisis y predicción de aceleraciones espectrales mediante aprendizaje residual GRU.

## Estructura

```text
app.py
model/
  train_model.py
  model.pkl              # generado después de entrenar
tabs/
  introduccion.py
  problema.py
  objetivos.py
  resultados.py
  prediccion.py
  limitaciones.py
  conclusiones.py
assets/
  styles.css
requirements.txt
```

## Datos esperados

Coloca estos archivos en la raíz del proyecto:

- `Resids_for_Eliasib.xlsx`
- `CopiaDataBaseSGC2.xlsx`

La base principal debe estar en formato largo con columnas como:

- `Record Sequence Number`
- `EQID_Code`
- `Station Code`
- `Hypocenter Depth (km)`
- `Magnitude`
- `Rrup_OpenQuake`
- `Cat` o `Soil_Class`
- `Tcorner`, `Tmax` o `T_max`
- `Period`
- `Sa`
- `Total`
- `Rvolc [km]`

## Ejecución

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python model/train_model.py --models nosam,nosam_elevation --epochs 60
python app.py
```

Para pruebas rápidas sin LOGO:

```bash
python model/train_model.py --models nosam --epochs 5 --skip-logo --verbose 1
```

Para ASK14:

```bash
pip install openquake.engine
python model/train_model.py --models ask14 --epochs 60
```
```
