from pathlib import Path
import sys
import argparse
import pickle
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tabs.decomp_variabilidad import (
    get_package,
    force_common_gate_object,
    nosam_long_residuals_from_excel,
    ask14_corrected_long_residuals,
    decompose_long_residuals,
)


CACHE_PATH = ROOT / "model" / "decomp_cache.pkl"


def make_key(model_name, apply_gate=False, alpha=None, beta=None):
    if model_name == "nosam":
        return "nosam"

    if model_name == "ask14_corrected" and not apply_gate:
        return "ask14_corrected__gate0"

    if model_name == "ask14_corrected" and apply_gate:
        return (
            "ask14_corrected__gate1__"
            f"alpha_{float(alpha):.4f}__"
            f"beta_{float(beta):.4f}"
        )

    raise ValueError(f"Combinación no reconocida: {model_name}, gate={apply_gate}")


def parse_values(text):
    """
    Acepta:
    - "1,2,3,4"
    - "1:5:0.5"  -> desde 1 hasta 5 con paso 0.5
    """
    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    if ":" in text:
        parts = [float(x.strip()) for x in text.split(":")]

        if len(parts) != 3:
            raise ValueError(
                "Formato inválido. Usa inicio:fin:paso. Ejemplo: 1:5:0.5"
            )

        start, stop, step = parts

        if step <= 0:
            raise ValueError("El paso debe ser mayor que 0.")

        values = np.arange(start, stop + 0.5 * step, step)
        return [round(float(v), 4) for v in values]

    return [round(float(x.strip()), 4) for x in text.split(",") if x.strip()]


def decomp_is_valid(df):
    """
    Evita aceptar en cache una descomposición vacía o dañada.
    """
    if df is None:
        return False

    if not hasattr(df, "columns"):
        return False

    required = ["Period", "tau", "phi", "sigma"]

    if any(c not in df.columns for c in required):
        return False

    if len(df) == 0:
        return False

    valid_count = (
        df[["tau", "phi", "sigma"]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
        .sum()
    )

    return valid_count > 0


def get_default_ask14_gate_params():
    package = get_package()

    if isinstance(package, Exception):
        raise RuntimeError(str(package))

    if "ask14" not in package.get("models", {}):
        raise ValueError("model.pkl no contiene el modelo ask14.")

    artifact = package["models"]["ask14"]
    gate = artifact.get("gate")

    if gate is None:
        return 3.0, 3.0

    gate = force_common_gate_object(gate)

    alpha = float(gate.get("alpha", 3.0))
    beta = float(gate.get("beta", 3.0))

    return alpha, beta


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    return {
        "created_at": None,
        "updated_at": None,
        "items": {},
        "metadata": {},
    }


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cache["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if cache.get("created_at") is None:
        cache["created_at"] = cache["updated_at"]

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"\nCache guardado en: {CACHE_PATH}")


def compute_nosam(cache, overwrite=False, min_records_per_period=8):
    key = make_key("nosam")

    if key in cache["items"] and not overwrite:
        old = cache["items"].get(key)

        if decomp_is_valid(old):
            print(f"[OK] NoSAm ya estaba precalculado y es válido: {key}")
            return cache

        print(
            f"[ADVERTENCIA] NoSAm estaba en cache pero estaba vacío/dañado. "
            f"Se recalculará: {key}"
        )

    print("\nCalculando descomposición NoSAm base...")

    long_df = nosam_long_residuals_from_excel()

    print(f"Registros largos NoSAm: {len(long_df)}")
    print(f"Períodos NoSAm detectados: {sorted(long_df['Period'].dropna().unique())}")

    decomp = decompose_long_residuals(
        long_df,
        min_records_per_period=min_records_per_period,
    )

    if not decomp_is_valid(decomp):
        print(decomp)
        raise RuntimeError(
            "La descomposición de NoSAm se calculó, pero no produjo valores válidos "
            "para tau, phi o sigma. Revisa columnas EQID_Code, Station Code, Period "
            "y Total en Resids_for_Eliasib.xlsx."
        )

    cache["items"][key] = decomp
    cache["metadata"][key] = {
        "model": "nosam",
        "apply_gate": False,
        "alpha": None,
        "beta": None,
        "n_periods": int(decomp["Period"].nunique()),
        "n_rows_long": int(len(long_df)),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[OK] NoSAm precalculado correctamente: {key}")
    return cache


def compute_ask14_no_gate(cache, overwrite=False, min_records_per_period=8):
    key = make_key("ask14_corrected", apply_gate=False)

    if key in cache["items"] and not overwrite:
        old = cache["items"].get(key)

        if decomp_is_valid(old):
            print(f"[OK] ASK14 corregido sin gate ya estaba precalculado y es válido: {key}")
            return cache

        print(
            f"[ADVERTENCIA] ASK14 sin gate estaba en cache pero estaba vacío/dañado. "
            f"Se recalculará: {key}"
        )

    print("\nCalculando descomposición ASK14 + GRU residual sin gate...")

    long_df = ask14_corrected_long_residuals(apply_gate=False)

    print(f"Registros largos ASK14 sin gate: {len(long_df)}")
    print(f"Períodos ASK14 sin gate detectados: {sorted(long_df['Period'].dropna().unique())}")

    decomp = decompose_long_residuals(
        long_df,
        min_records_per_period=min_records_per_period,
    )

    if not decomp_is_valid(decomp):
        print(decomp)
        raise RuntimeError(
            "La descomposición de ASK14 sin gate se calculó, pero no produjo valores válidos."
        )

    cache["items"][key] = decomp
    cache["metadata"][key] = {
        "model": "ask14_corrected",
        "apply_gate": False,
        "alpha": None,
        "beta": None,
        "n_periods": int(decomp["Period"].nunique()),
        "n_rows_long": int(len(long_df)),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[OK] ASK14 sin gate precalculado correctamente: {key}")
    return cache


def compute_ask14_with_gate(
    cache,
    alpha,
    beta,
    overwrite=False,
    min_records_per_period=8,
):
    key = make_key(
        "ask14_corrected",
        apply_gate=True,
        alpha=alpha,
        beta=beta,
    )

    if key in cache["items"] and not overwrite:
        old = cache["items"].get(key)

        if decomp_is_valid(old):
            print(f"[OK] ASK14 con gate ya estaba precalculado y es válido: {key}")
            return cache

        print(
            f"[ADVERTENCIA] ASK14 con gate estaba en cache pero estaba vacío/dañado. "
            f"Se recalculará: {key}"
        )

    print(
        "\nCalculando descomposición ASK14 + GRU residual "
        f"con gate alpha={alpha:.4f}, beta={beta:.4f}..."
    )

    long_df = ask14_corrected_long_residuals(
        apply_gate=True,
        alpha=float(alpha),
        beta=float(beta),
    )

    print(f"Registros largos ASK14 con gate: {len(long_df)}")
    print(f"Períodos ASK14 con gate detectados: {sorted(long_df['Period'].dropna().unique())}")

    decomp = decompose_long_residuals(
        long_df,
        min_records_per_period=min_records_per_period,
    )

    if not decomp_is_valid(decomp):
        print(decomp)
        raise RuntimeError(
            f"La descomposición de ASK14 con gate alpha={alpha:.4f}, beta={beta:.4f} "
            "se calculó, pero no produjo valores válidos."
        )

    cache["items"][key] = decomp
    cache["metadata"][key] = {
        "model": "ask14_corrected",
        "apply_gate": True,
        "alpha": float(alpha),
        "beta": float(beta),
        "n_periods": int(decomp["Period"].nunique()),
        "n_rows_long": int(len(long_df)),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[OK] ASK14 con gate precalculado correctamente: {key}")
    return cache


def format_timedelta(delta):
    seconds = int(delta.total_seconds())
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}min {secs}s"

    if minutes > 0:
        return f"{minutes}min {secs}s"

    return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Precalcula la descomposición de variabilidad para Dash."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha único del gate. Si no se pasa, usa el alpha del model.pkl.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta único del gate. Si no se pasa, usa el beta del model.pkl.",
    )

    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help=(
            "Lista o rango de alphas. Ejemplos: "
            "--alphas 1,2,3,4 o --alphas 1:5:0.5"
        ),
    )

    parser.add_argument(
        "--betas",
        type=str,
        default=None,
        help=(
            "Lista o rango de betas. Ejemplos: "
            "--betas 2,3,4 o --betas 2:5:0.5"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalcula aunque ya exista en model/decomp_cache.pkl.",
    )

    parser.add_argument(
        "--min-records-per-period",
        type=int,
        default=8,
        help="Mínimo de registros por período para ajustar MixedLM.",
    )

    args = parser.parse_args()

    default_alpha, default_beta = get_default_ask14_gate_params()

    alphas = parse_values(args.alphas)
    betas = parse_values(args.betas)

    if alphas is None:
        if args.alpha is not None:
            alphas = [round(float(args.alpha), 4)]
        else:
            alphas = [round(float(default_alpha), 4)]

    if betas is None:
        if args.beta is not None:
            betas = [round(float(args.beta), 4)]
        else:
            betas = [round(float(default_beta), 4)]

    total_combos = len(alphas) * len(betas)

    print("==============================================")
    print(" PRECOMPUTE DESCOMPOSICIÓN DE VARIABILIDAD")
    print("==============================================")
    print(f"alphas usados      : {alphas}")
    print(f"betas usados       : {betas}")
    print(f"combinaciones gate : {total_combos}")
    print(f"overwrite          : {args.overwrite}")
    print(f"min records/period : {args.min_records_per_period}")
    print(f"cache              : {CACHE_PATH}")
    print("==============================================")

    cache = load_cache()

    cache = compute_nosam(
        cache,
        overwrite=args.overwrite,
        min_records_per_period=args.min_records_per_period,
    )

    cache = compute_ask14_no_gate(
        cache,
        overwrite=args.overwrite,
        min_records_per_period=args.min_records_per_period,
    )

    combo_idx = 0
    start_time = datetime.now()

    for alpha in alphas:
        for beta in betas:
            combo_idx += 1

            elapsed = datetime.now() - start_time
            elapsed_seconds = max(elapsed.total_seconds(), 1e-9)

            avg_seconds_per_combo = elapsed_seconds / max(combo_idx - 1, 1)
            remaining_combos = total_combos - combo_idx + 1
            eta_seconds = avg_seconds_per_combo * remaining_combos

            eta_delta = datetime.fromtimestamp(0) - datetime.fromtimestamp(0)
            eta_delta = eta_delta + np.timedelta64(int(eta_seconds), "s").astype("timedelta64[s]").astype(object)

            print("\n" + "=" * 70)
            print(
                f"PROGRESO GATE: {combo_idx}/{total_combos} "
                f"({100.0 * combo_idx / total_combos:.1f}%)"
            )
            print(f"alpha = {float(alpha):.4f} | beta = {float(beta):.4f}")
            print(f"tiempo transcurrido = {format_timedelta(elapsed)}")

            if combo_idx > 1:
                print(f"tiempo estimado restante = {format_timedelta(eta_delta)}")
            else:
                print("tiempo estimado restante = calculando...")

            print("=" * 70)

            cache = compute_ask14_with_gate(
                cache,
                alpha=float(alpha),
                beta=float(beta),
                overwrite=args.overwrite,
                min_records_per_period=args.min_records_per_period,
            )

            save_cache(cache)

    print("\n==============================================")
    print(" LISTO")
    print("==============================================")
    print("Dash cargará la descomposición desde:")
    print(CACHE_PATH)
    print("\nClaves guardadas:")

    for key in cache.get("items", {}).keys():
        print(f" - {key}")


if __name__ == "__main__":
    main()