from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "model" / "decomp_cache.pkl"
OUT_PATH = ROOT / "model" / "decomp_sigma_filter.csv"


# Cambia este umbral si quieres ser más o menos estricto
THRESHOLD = -5.0


# Etiquetas esperadas para tus betas actuales
BETA_LABELS = {
    2.7157: "P95",
    2.9785: "P99",
    3.3804: "P100",
    3.5494: "P105",
    3.7184: "P110",
}


def nearest_beta_label(beta):
    beta = float(beta)

    if not BETA_LABELS:
        return ""

    nearest = min(BETA_LABELS.keys(), key=lambda x: abs(x - beta))

    if abs(nearest - beta) <= 5e-4:
        return BETA_LABELS[nearest]

    return ""


def mean_reduction(base_df, model_df, metric):
    merged = base_df[["Period", metric]].merge(
        model_df[["Period", metric]],
        on="Period",
        suffixes=("_nosam", "_model"),
    )

    base = merged[f"{metric}_nosam"].to_numpy(dtype=float)
    model = merged[f"{metric}_model"].to_numpy(dtype=float)

    valid = np.isfinite(base) & np.isfinite(model) & (base > 0)

    if not valid.any():
        return np.nan

    reduction = 100.0 * (base[valid] - model[valid]) / base[valid]

    return float(np.nanmean(reduction))


def main():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No existe {CACHE_PATH}. Primero corre model/precompute_decomp.py."
        )

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    items = cache.get("items", {})

    if "nosam" not in items:
        raise KeyError("El cache no contiene la descomposición base de NoSAm.")

    nosam = items["nosam"]

    rows = []

    # Caso sin gate
    if "ask14_corrected__gate0" in items:
        ask14_no_gate = items["ask14_corrected__gate0"]

        rows.append(
            {
                "case": "ASK14 + GRU residual sin gate",
                "apply_gate": False,
                "alpha": np.nan,
                "beta": np.nan,
                "beta_label": "sin gate",
                "sigma_red_mean": mean_reduction(nosam, ask14_no_gate, "sigma"),
                "tau_red_mean": mean_reduction(nosam, ask14_no_gate, "tau"),
                "phi_red_mean": mean_reduction(nosam, ask14_no_gate, "phi"),
                "cache_key": "ask14_corrected__gate0",
            }
        )

    pattern = re.compile(
        r"^ask14_corrected__gate1__alpha_([0-9.+-eE]+)__beta_([0-9.+-eE]+)$"
    )

    for key, df in items.items():
        match = pattern.match(str(key))

        if not match:
            continue

        alpha = float(match.group(1))
        beta = float(match.group(2))

        rows.append(
            {
                "case": "ASK14 + GRU residual con gate",
                "apply_gate": True,
                "alpha": alpha,
                "beta": beta,
                "beta_label": nearest_beta_label(beta),
                "sigma_red_mean": mean_reduction(nosam, df, "sigma"),
                "tau_red_mean": mean_reduction(nosam, df, "tau"),
                "phi_red_mean": mean_reduction(nosam, df, "phi"),
                "cache_key": key,
            }
        )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError("No encontré combinaciones ASK14 en decomp_cache.pkl.")

    results = results.sort_values(
        by=["sigma_red_mean", "tau_red_mean", "phi_red_mean"],
        ascending=[False, False, False],
    )

    filtered = results[results["sigma_red_mean"] > THRESHOLD].copy()

    filtered.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("====================================================")
    print("FILTRO DE COMBINACIONES GATE")
    print("====================================================")
    print(f"Umbral usado: sigma_red_mean > {THRESHOLD:.2f}%")
    print(f"Total combinaciones evaluadas: {len(results)}")
    print(f"Combinaciones aceptadas: {len(filtered)}")
    print(f"Archivo guardado en: {OUT_PATH}")
    print("====================================================\n")

    if filtered.empty:
        print("No hay combinaciones que cumplan el criterio.")
        print("\nMejores 15 combinaciones encontradas:")
        print(
            results[
                [
                    "case",
                    "alpha",
                    "beta",
                    "beta_label",
                    "sigma_red_mean",
                    "tau_red_mean",
                    "phi_red_mean",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )
        return

    print("Combinaciones con sigma_red_mean > -5%, ordenadas de mejor a peor:\n")

    print(
        filtered[
            [
                "case",
                "alpha",
                "beta",
                "beta_label",
                "sigma_red_mean",
                "tau_red_mean",
                "phi_red_mean",
            ]
        ].to_string(index=False)
    )

    print("\n====================================================")
    print("TOP 10 RECOMENDADAS")
    print("====================================================")
    print(
        filtered[
            [
                "alpha",
                "beta",
                "beta_label",
                "sigma_red_mean",
                "tau_red_mean",
                "phi_red_mean",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()