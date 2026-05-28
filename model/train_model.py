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
            "No se pudo importar OpenQuake/ASK14. "
            f"Error original: {type(exc).__name__}: {exc}"
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
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
# CURVAS DE EVALUACIÓN: BASE, OOF, OOF FILTRADO Y FULL
# =========================================================

MIN_N_VALID_FILTER_DEFAULT = 20


def _valid_event_values_from_fold_summary(
    fold_summary: Optional[pd.DataFrame],
    min_n_valid: int,
) -> List[str]:
    """Extrae eventos/folds con suficientes registros de validación."""
    if fold_summary is None or len(fold_summary) == 0:
        return []

    df = fold_summary.copy()

    event_col = "event" if "event" in df.columns else "group_left_out"
    n_col = "n_valid" if "n_valid" in df.columns else "n_va"

    if event_col not in df.columns or n_col not in df.columns:
        return []

    return (
        df.loc[df[n_col].astype(float) >= float(min_n_valid), event_col]
        .astype(str)
        .tolist()
    )


def compute_comparison_curves(
    periods: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    Yhat_oof: Optional[np.ndarray],
    Yhat_full: Optional[np.ndarray],
    df_wide: pd.DataFrame,
    fold_summary: Optional[pd.DataFrame],
    min_n_valid_filter: int = MIN_N_VALID_FILTER_DEFAULT,
    metrics: Tuple[str, ...] = ("SD", "RMSE", "MAE", "MSE"),
) -> Dict[str, pd.DataFrame]:
    """
    Calcula curvas equivalentes a los notebooks:
    - residual base de la GMPE;
    - residual corregido OOF con todos los folds;
    - residual corregido OOF filtrado por eventos con n_valid >= min_n_valid_filter;
    - residual corregido full train.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)

    out: Dict[str, pd.DataFrame] = {}

    valid_events = _valid_event_values_from_fold_summary(
        fold_summary,
        min_n_valid_filter,
    )

    if EVENT_COL in df_wide.columns and len(valid_events) > 0:
        event_values = df_wide[EVENT_COL].astype(str).to_numpy()
        row_filter = np.isin(event_values, valid_events)
    else:
        row_filter = np.zeros(W.shape[0], dtype=bool)

    W_filtered = W.copy()
    W_filtered[~row_filter, :] = 0.0

    for metric in metrics:
        metric = metric.upper()

        base_all = metric_curve(Y, W, metric)

        oof_all = np.full_like(base_all, np.nan, dtype=float)
        oof_filtered = np.full_like(base_all, np.nan, dtype=float)
        full = np.full_like(base_all, np.nan, dtype=float)

        if Yhat_oof is not None:
            Yhat_oof_arr = np.asarray(Yhat_oof, dtype=float)

            eps_oof = np.where(W > 0, Y - Yhat_oof_arr, np.nan)
            oof_all = metric_curve(eps_oof, W, metric)

            eps_oof_filt = np.where(W_filtered > 0, Y - Yhat_oof_arr, np.nan)
            oof_filtered = metric_curve(eps_oof_filt, W_filtered, metric)

        if Yhat_full is not None:
            Yhat_full_arr = np.asarray(Yhat_full, dtype=float)
            eps_full = np.where(W > 0, Y - Yhat_full_arr, np.nan)
            full = metric_curve(eps_full, W, metric)

        df_metric = pd.DataFrame(
            {
                "Period": np.asarray(periods, dtype=float),
                f"{metric}_base": base_all,
                f"{metric}_model_oof_all": oof_all,
                f"{metric}_model_oof_filtered_nva{min_n_valid_filter}": oof_filtered,
                f"{metric}_model_full_train": full,
            }
        )

        for col in [
            f"{metric}_model_oof_all",
            f"{metric}_model_oof_filtered_nva{min_n_valid_filter}",
            f"{metric}_model_full_train",
        ]:
            suffix = col.replace(f"{metric}_model_", "")

            df_metric[f"{metric}_reduction_pct_{suffix}"] = np.where(
                df_metric[f"{metric}_base"] > 0,
                100.0 * (df_metric[f"{metric}_base"] - df_metric[col])
                / df_metric[f"{metric}_base"],
                np.nan,
            )

        out[metric] = df_metric

    return out


# =========================================================
# GATE EUCLIDIANO POST HOC
# =========================================================

DEFAULT_SOIL_CONF = 1.0
GATE_ALPHA = 3.0
GATE_BETA_Q = 0.95


COMMON_GATE_COLS = ["Rrup_km", "Magnitude", "Hypocenter Depth (km)"]


def gate_feature_columns_for_model(model_key: str) -> List[str]:
    """
    Variables comunes usadas por el gate post hoc en todos los modelos.

    Rvolc y Station Elevation pueden entrar a la GMPE o a la GRU residual
    cuando el escenario lo requiera, pero no entran al cálculo del gate.
    """
    return COMMON_GATE_COLS.copy()


def get_gate_feature_df(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Construye variables físicas para el gate, revirtiendo transformaciones logarítmicas cuando aplica."""

    if RRUP_RAW_COL in df.columns:
        rrup_km = df[RRUP_RAW_COL].to_numpy(dtype=float)
    else:
        rrup_km = np.exp(df["Rrup_OpenQuake"].to_numpy(dtype=float))

    out = pd.DataFrame(
        {
            "Rrup_km": rrup_km,
            "Magnitude": df["Magnitude"].to_numpy(dtype=float),
            "Hypocenter Depth (km)": df["Hypocenter Depth (km)"].to_numpy(dtype=float),
            SOIL_COL: df[SOIL_COL].to_numpy(),
        }
    )

    if model_key in ["nosam", "nosam_elevation"]:
        if RVOLC_RAW_COL in df.columns:
            out["Rvolc_km"] = df[RVOLC_RAW_COL].to_numpy(dtype=float)
        elif RVOLC_COL in df.columns:
            out["Rvolc_km"] = np.expm1(df[RVOLC_COL].to_numpy(dtype=float))
        else:
            out["Rvolc_km"] = 0.0

    return out


def fit_euclidean_gate(
    df_train: pd.DataFrame,
    model_key: str,
    alpha: float = GATE_ALPHA,
    beta_q: float = GATE_BETA_Q,
    soil_conf_map: Optional[dict] = None,
) -> Dict[str, Any]:
    if soil_conf_map is None:
        soil_conf_map = {}

    gate_df = get_gate_feature_df(df_train, model_key=model_key)
    gate_cols = gate_feature_columns_for_model(model_key)

    X_gate = gate_df[gate_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_gate_std = scaler.fit_transform(X_gate)

    d_train = np.linalg.norm(X_gate_std, axis=1)
    beta = float(np.quantile(d_train, beta_q))

    return {
        "model_key": model_key,
        "scaler": scaler,
        "alpha": float(alpha),
        "beta": beta,
        "beta_q": float(beta_q),
        "soil_conf_map": soil_conf_map,
        "gate_cont_cols": gate_cols,
        "train_distance_median": float(np.nanmedian(d_train)),
        "train_distance_p90": float(np.nanquantile(d_train, 0.90)),
        "train_distance_p95": float(np.nanquantile(d_train, 0.95)),
    }


def physical_scenario_frame(
    model_key: str,
    magnitude: float,
    rrup_km: float,
    zhypo_km: float,
    soil_class: int,
    rvolc_km: float = 0.0,
) -> pd.DataFrame:
    row = {
        "Rrup_km": float(rrup_km),
        "Magnitude": float(magnitude),
        "Hypocenter Depth (km)": float(zhypo_km),
        SOIL_COL: int(soil_class),
    }

    if model_key in ["nosam", "nosam_elevation"]:
        row["Rvolc_km"] = float(rvolc_km)

    return pd.DataFrame([row])


def _subset_standard_scaler(scaler, old_cols, new_cols):
    """
    Recorta un StandardScaler ya entrenado para usar solamente un subconjunto
    de variables. Esto permite usar un model.pkl viejo aunque el gate original
    haya sido entrenado con Rvolc.
    """
    from copy import deepcopy

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


def force_common_gate_object(gate_obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Fuerza cualquier gate existente a usar únicamente:
    Rrup_km, Magnitude y Hypocenter Depth (km).

    Esto evita que un model.pkl antiguo use Rvolc dentro del gate.
    """
    if gate_obj is None:
        return None

    from copy import deepcopy

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

        if "beta" in gate and not gate.get("beta_manual", False):
            gate["beta"] = float(gate["beta"]) * np.sqrt(len(new_cols) / len(old_cols))

    gate["gate_cont_cols"] = new_cols
    gate["original_gate_cont_cols"] = old_cols
    gate["uses_common_gate_only"] = True

    return gate


def apply_euclidean_gate_from_physical(
    physical_df: pd.DataFrame,
    gate_obj: Dict[str, Any],
    periods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el gate usando solamente las variables comunes:
    Rrup_km, Magnitude y Hypocenter Depth (km).

    Aunque physical_df tenga Rvolc_km o Station Elevation, esas columnas
    no participan en el cálculo del gate.
    """
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

    g_dist = 1.0 / (1.0 + np.exp(gate_obj["alpha"] * (d - gate_obj["beta"])))

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


def gate_weight_single(
    gate_obj: Optional[Dict[str, Any]],
    model_key: str,
    periods: np.ndarray,
    magnitude: float,
    rrup_km: float,
    zhypo_km: float,
    soil_class: int,
    rvolc_km: float = 0.0,
) -> float:
    if gate_obj is None:
        return 1.0

    physical_df = physical_scenario_frame(
        model_key=model_key,
        magnitude=magnitude,
        rrup_km=rrup_km,
        zhypo_km=zhypo_km,
        soil_class=soil_class,
        rvolc_km=rvolc_km,
    )

    _, g_scalar, _, _, _ = apply_euclidean_gate_from_physical(
        physical_df,
        gate_obj,
        periods,
    )

    return float(g_scalar[0])


def apply_posthoc_gate_full(
    df_wide: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    Yhat_raw_full: np.ndarray,
    periods: np.ndarray,
    model_key: str,
    gate_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if gate_obj is None:
        gate_obj = fit_euclidean_gate(df_wide, model_key=model_key)

    gate_df = get_gate_feature_df(df_wide, model_key=model_key)

    G_full, g_full, gdist_full, soilconf_full, d_full = apply_euclidean_gate_from_physical(
        gate_df,
        gate_obj,
        periods,
    )

    Yhat_gated_full = np.asarray(Yhat_raw_full, dtype=float) * G_full
    eps_corr_gated_full = np.where(W > 0, np.asarray(Y, dtype=float) - Yhat_gated_full, np.nan)

    return {
        "Yhat_gated_full": Yhat_gated_full,
        "G_full": G_full,
        "g_scalar_full": g_full,
        "g_dist_full": gdist_full,
        "soil_conf_full": soilconf_full,
        "d_full": d_full,
        "eps_corr_gated_full": eps_corr_gated_full,
        "sd_model_gated_full": sd_curve(eps_corr_gated_full, W),
    }


# =========================================================
# GMPE BASE PARA PREDICCIÓN FÍSICA
# =========================================================

def nosam_sa_spectrum(
    periods,
    magnitude,
    rrup_km,
    zhypo_km,
    soil_class,
    rvolc_km=0.0,
) -> np.ndarray:
    """Devuelve Sa lineal de NoSAm. Requiere NoSAm_GMMs_2024.py en la raíz del proyecto."""

    try:
        from dash_espectros_residuales.NoSAm_GMMs_2024 import NoSAm_Crustal_2023
    except Exception as exc:
        raise ImportError(
            "No se pudo importar NoSAm_GMMs_2024.NoSAm_Crustal_2023. "
            "Coloca NoSAm_GMMs_2024.py en la raíz del proyecto. "
            f"Error original: {type(exc).__name__}: {exc}"
        ) from exc

    sa = np.empty(len(periods), dtype=float)

    for j, t in enumerate(np.asarray(periods, dtype=float)):
        result = NoSAm_Crustal_2023(
            float(t),
            float(magnitude),
            float(rrup_km),
            int(soil_class),
            "average",
            float(zhypo_km),
            float(rvolc_km),
        )

        sa[j] = float(np.asarray(result).reshape(-1)[0])

    return sa


def base_sa_spectrum(
    model_key: str,
    periods: np.ndarray,
    magnitude: float,
    rrup_km: float,
    zhypo_km: float,
    soil_class: int,
    rvolc_km: float = 0.0,
) -> Tuple[np.ndarray, str]:
    if model_key in ["nosam", "nosam_elevation"]:
        return (
            nosam_sa_spectrum(
                periods=periods,
                magnitude=magnitude,
                rrup_km=rrup_km,
                zhypo_km=zhypo_km,
                soil_class=soil_class,
                rvolc_km=rvolc_km,
            ),
            "NoSAm",
        )

    if model_key == "ask14":
        vs30 = soil_class_to_vs30(int(soil_class))

        ln_base = ask14_ln_spectrum(
            periods=periods,
            Mag=float(magnitude),
            Rrup=float(rrup_km),
            Zhypo=float(zhypo_km),
            Vs30_Val=vs30,
        )

        return np.exp(ln_base), "ASK14"

    raise ValueError(f"Modelo no reconocido: {model_key}")
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

    yhat_oof = logo_artifact["Yhat_oof"] if logo_artifact else None
    fold_summary = logo_artifact["fold_summary"] if logo_artifact else None

    evaluation_curves = compute_comparison_curves(
        periods=periods,
        Y=Y,
        W=W,
        Yhat_oof=yhat_oof,
        Yhat_full=full_artifact["Yhat_full"],
        df_wide=df_wide,
        fold_summary=fold_summary,
        min_n_valid_filter=MIN_N_VALID_FILTER_DEFAULT,
    )

    gate_obj = fit_euclidean_gate(
        df_train=df_wide,
        model_key=model_key,
        alpha=GATE_ALPHA,
        beta_q=GATE_BETA_Q,
        soil_conf_map={},
    )

    gate_full = apply_posthoc_gate_full(
        df_wide=df_wide,
        Y=Y,
        W=W,
        Yhat_raw_full=full_artifact["Yhat_full"],
        periods=periods,
        model_key=model_key,
        gate_obj=gate_obj,
    )

    meta_cols = [
        c
        for c in [
            RECORD_COL,
            EVENT_COL,
            STATION_COL,
            "Hypocenter Depth (km)",
            "Magnitude",
            "Rrup_OpenQuake",
            RRUP_RAW_COL,
            RVOLC_COL,
            RVOLC_RAW_COL,
            SOIL_COL,
            TMAX_COL,
            ELEV_COL,
            LOG_ELEV_COL,
        ]
        if c in df_wide.columns
    ]

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
        "evaluation_curves": evaluation_curves,
        "min_n_valid_filter": MIN_N_VALID_FILTER_DEFAULT,
        "fold_summary": fold_summary,
        "Yhat_oof": yhat_oof,
        "Yhat_full": full_artifact["Yhat_full"],
        "Y": Y,
        "W": W,
        "Sa_raw": Sa_raw,
        "gate": gate_obj,
        "gate_full": gate_full,
        "df_wide_metadata": df_wide[meta_cols].copy(),
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
            from dash_espectros_residuales.NoSAm_GMMs_2024 import NoSAm_Crustal_2023

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
        default="nosam,nosam_elevation,ask14",
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