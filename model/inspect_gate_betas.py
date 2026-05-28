from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.train_model import load_model_package


def pick_col(df, candidates):
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"No encontré ninguna de estas columnas: {candidates}\n"
        f"Columnas disponibles: {list(df.columns)}"
    )


def main():
    package = load_model_package("model/model.pkl")
    art = package["models"]["ask14"]
    gate = art["gate"]

    print("====================================")
    print("GATE GUARDADO EN model.pkl")
    print("====================================")
    print("gate_cont_cols =", gate.get("gate_cont_cols"))
    print("alpha guardado =", gate.get("alpha"))
    print("beta guardado  =", gate.get("beta"))
    print("scaler mean    =", gate["scaler"].mean_)
    print("scaler scale   =", gate["scaler"].scale_)

    path = ROOT / "Resids_for_Eliasib.xlsx"

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    raw = pd.read_excel(path)

    mag_col = pick_col(raw, ["Magnitude"])
    rrup_col = pick_col(raw, ["Rrup_OpenQuake", "Rrup_km", "Rrup (km)", "Rrup"])
    zhypo_col = pick_col(raw, ["Hypocenter Depth (km)", "Hypocentral Depth (km)", "Zhypo", "Depth"])
    eqid_col = pick_col(raw, ["EQID_Code", "EQID"])
    station_col = pick_col(raw, ["Station Code", "Station", "Station_Code"])

    # Una sola fila por registro evento-estación.
    # Esto evita que el formato largo duplique registros por período.
    df = raw[[eqid_col, station_col, mag_col, rrup_col, zhypo_col]].copy()
    df = df.drop_duplicates(subset=[eqid_col, station_col])
    df = df.rename(
        columns={
            mag_col: "Magnitude",
            rrup_col: "Rrup_km",
            zhypo_col: "Hypocenter Depth (km)",
        }
    )

    for c in ["Magnitude", "Rrup_km", "Hypocenter Depth (km)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Magnitude", "Rrup_km", "Hypocenter Depth (km)"])

    # Asegurar Rrup físico. Si por alguna razón viniera logarítmico, se avisa.
    if df["Rrup_km"].median() < 10:
        print("\n[ADVERTENCIA] Rrup_km parece estar en escala logarítmica por su mediana.")
        print(df["Rrup_km"].describe())

    X = df[gate["gate_cont_cols"]].to_numpy(float)
    Xs = gate["scaler"].transform(X)
    d = np.linalg.norm(Xs, axis=1)

    p50 = np.percentile(d, 50)
    p80 = np.percentile(d, 80)
    p95 = np.percentile(d, 95)
    p99 = np.percentile(d, 99)
    p100 = np.percentile(d, 100)
    p105 = 1.05 * p100

    print("\n====================================")
    print("DISTANCIAS RECALCULADAS DESDE EXCEL")
    print("Una fila por EQID_Code + Station Code")
    print("====================================")
    print(f"n registros únicos = {len(df)}")
    print(f"P50  = {p50:.4f}")
    print(f"P80  = {p80:.4f}")
    print(f"P95  = {p95:.4f}")
    print(f"P99  = {p99:.4f}")
    print(f"P100 = {p100:.4f}")
    print(f"P105 = {p105:.4f}  # 1.05 * P100")

    print("\n====================================")
    print("COMPARACIÓN")
    print("====================================")
    print(f"beta guardado en model.pkl = {float(gate.get('beta')):.4f}")
    print(f"P95 recalculado desde Excel = {p95:.4f}")
    print(f"diferencia absoluta         = {abs(float(gate.get('beta')) - p95):.6f}")

    print("\n====================================")
    print("Línea recomendada para precompute")
    print("====================================")
    print(",".join(f"{x:.4f}" for x in [p95, p99, p100, p105]))


if __name__ == "__main__":
    main()