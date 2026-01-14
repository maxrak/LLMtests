import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "dati/dati_irius.csv"   # cambia se necessario
#CSV_PATH = "dati/threat_metrics.csv"   # cambia se necessario
IMG_DIR = "imgs"
RESULTS_DIR = "results"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_kde(values: np.ndarray, num_points: int = 500):
    """
    Calcola una KDE gaussiana usando la regola di Silverman per il bandwidth.
    Ritorna xs, ys.
    """
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        raise ValueError("Servono almeno 2 valori per calcolare la KDE.")
    std = np.std(values)
    if std == 0:
        raise ValueError("Deviazione standard nulla: KDE non definita.")

    # Bandwidth di Silverman
    h = 1.06 * std * n ** (-1 / 5)

    xs = np.linspace(values.min(), values.max(), num_points)
    ys = np.zeros_like(xs)

    for v in values:
        ys += np.exp(-0.5 * ((xs - v) / h) ** 2) / (np.sqrt(2 * np.pi) * h)
    ys /= n

    return xs, ys


def plot_pdf_with_kde(df: pd.DataFrame, column: str, bins: int = 30,
                      output_dir: str = IMG_DIR):
    """
    Crea istogramma normalizzato (PDF empirica) + KDE per `column` e salva
    il grafico in imgs/.
    """
    if column not in df.columns:
        print(f"Colonna '{column}' non trovata, salto PDF+KDE.")
        return

    vals = df[column].dropna().values
    if len(vals) == 0:
        print(f"Nessun valore per '{column}', salto PDF+KDE.")
        return

    ensure_dir(output_dir)

    xs, ys = compute_kde(vals)

    plt.figure()
    plt.hist(vals, bins=bins, density=True, alpha=0.5)  # PDF empirica
    plt.plot(xs, ys)  # KDE

    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"PDF + KDE of {column}")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{column}_pdf_kde.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Salvato: {output_path}")


def boxplot_global_and_by_app(df: pd.DataFrame, column: str,
                              output_dir: str = IMG_DIR):
    if column not in df.columns:
        print(f"Colonna '{column}' non trovata, salto boxplot.")
        return
    if "app_id" not in df.columns:
        print("Colonna 'app_id' non trovata, impossibile fare boxplot per app_id.")
        return

    ensure_dir(output_dir)

    vals_global = df[column].dropna()
    if len(vals_global) == 0:
        print(f"Nessun valore per '{column}', salto boxplot.")
        return

    data = [vals_global]
    labels = ["global"]

    app_ids = sorted(df["app_id"].dropna().unique())

    for app in app_ids:
        sub_vals = df.loc[df["app_id"] == app, column].dropna().values
        if len(sub_vals) == 0:
            continue
        data.append(sub_vals)
        labels.append(str(app))

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels, showmeans=True)

    # <<< Qui ruotiamo le label >>>
    plt.xticks(rotation=45)  # oppure rotation=90 se vuoi verticale

    plt.xlabel("global / app_id")
    plt.ylabel(column)
    plt.title(f"Boxplot of {column} (global and by app_id)")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{column}_boxplot_global_and_apps.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Salvato: {output_path}")

def reliability_label(n: int) -> str:
    """
    Etichetta qualitativa di affidabilità in base alla numerosità campionaria.
    """
    if n >= 100:
        return "alta"
    elif n >= 30:
        return "moderata"
    else:
        return "bassa"


def compute_stats_to_csv(df: pd.DataFrame,
                         columns=("precision", "recall"),
                         output_dir: str = RESULTS_DIR,
                         filename: str = "metrics_stats.csv"):
    """
    Calcola statistiche per ogni metrica e per:
      - globale
      - ciascun app_id
    e salva tutto in un CSV nella cartella results.
    Statistiche: n, mean, var, coeff. di variazione, errore standard,
    half-width dell'IC95%, etichetta di affidabilità.
    """
    ensure_dir(output_dir)
    rows = []

    if "app_id" in df.columns:
        app_ids = sorted(df["app_id"].dropna().unique())
    else:
        app_ids = []

    for col in columns:
        if col not in df.columns:
            print(f"Colonna '{col}' non trovata, salto nelle statistiche.")
            continue

        # Global
        vals_global = df[col].dropna().values
        if len(vals_global) > 0:
            n = len(vals_global)
            mean = vals_global.mean()
            var = vals_global.var(ddof=1) if n > 1 else 0.0
            std = np.sqrt(var)
            cv = std / mean if mean != 0 else np.nan
            se = std / np.sqrt(n) if n > 0 else np.nan
            ci95 = 1.96 * se if n > 1 else np.nan
            rel = reliability_label(n)

            rows.append({
                "metric": col,
                "scope_type": "global",
                "scope_value": "global",
                "n": n,
                "mean": mean,
                "variance": var,
                "std_dev": std,
                "coef_variation": cv,
                "std_error": se,
                "ci95_half_width": ci95,
                "reliability": rel
            })

        # Per app_id
        for app in app_ids:
            sub_vals = df[df["app_id"] == app][col].dropna().values
            if len(sub_vals) == 0:
                continue
            n = len(sub_vals)
            mean = sub_vals.mean()
            var = sub_vals.var(ddof=1) if n > 1 else 0.0
            std = np.sqrt(var)
            cv = std / mean if mean != 0 else np.nan
            se = std / np.sqrt(n) if n > 0 else np.nan
            ci95 = 1.96 * se if n > 1 else np.nan
            rel = reliability_label(n)

            rows.append({
                "metric": col,
                "scope_type": "app_id",
                "scope_value": app,
                "n": n,
                "mean": mean,
                "variance": var,
                "std_dev": std,
                "coef_variation": cv,
                "std_error": se,
                "ci95_half_width": ci95,
                "reliability": rel
            })

    if not rows:
        print("Nessuna statistica calcolata, CSV non creato.")
        return

    stats_df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, filename)
    stats_df.to_csv(output_path, index=False)
    print(f"Statistiche salvate in: {output_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    # PDF + KDE per precision e recall
    for col in ("precision", "recall"):
        plot_pdf_with_kde(df, column=col, bins=30, output_dir=IMG_DIR)

    # Boxplot unici (globali + per app_id) per precision e recall
    for col in ("precision", "recall"):
        boxplot_global_and_by_app(df, column=col, output_dir=IMG_DIR)

    # Statistiche in CSV (globali e per app_id)
    compute_stats_to_csv(df, columns=("precision", "recall"),
                         output_dir=RESULTS_DIR,
                         filename="metrics_stats.csv")


if __name__ == "__main__":
    main()