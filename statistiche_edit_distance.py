import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def summarize_metric(group, metric, confidence=0.95):
    """
    Calcola statistiche descrittive e intervalli di confidenza per una metrica.

    Parametri
    ---------
    group : pandas.DataFrame
        Sottinsieme del dataframe su cui calcolare le statistiche.
    metric : str
        Nome della colonna (metrica) su cui lavorare.
    confidence : float
        Livello di confidenza per l'intervallo (default 0.95 per il 95%).

    Ritorna
    -------
    dict
        Dizionario con media, deviazione standard, coefficiente di variazione,
        errore standard, intervallo di confidenza inferiore e superiore, n.
    """
    data = group[metric].dropna().to_numpy()
    n = len(data)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "conf_level": confidence,
        }

    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0

    # Intervallo di confidenza con distribuzione t di Student
    alpha = 1 - confidence
    if n > 1:
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * se
    else:
        # Con un solo dato non ha molto senso parlare di IC: mettiamo margin=0
        t_crit = np.nan
        margin = 0.0

    ci_low = mean - margin
    ci_high = mean + margin

    cv = (std / mean * 100) if mean != 0 else np.nan

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": cv,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "conf_level": confidence,
    }


def compute_statistics(csv_path, confidence=0.95):
    """
    Legge il file CSV e calcola:
    - statistiche globali
    - statistiche per combinazione (case, version)

    per le metriche:
    - graph_edit_distance
    - validity_edit_distance
    """
    df = pd.read_csv(csv_path)

    metrics = ["graph_edit_distance", "validity_edit_distance"]
    results_global = {}
    results_grouped = {}

    # Statistiche globali
    for m in metrics:
        results_global[m] = summarize_metric(df, m, confidence=confidence)

    # Statistiche per combinazione (case, version)
    grouped = df.groupby(["case", "version"], dropna=False)
    for (case, version), group in grouped:
        key = (case, version)
        results_grouped[key] = {}
        for m in metrics:
            results_grouped[key][m] = summarize_metric(group, m, confidence=confidence)

    return results_global, results_grouped


def print_results(results_global, results_grouped):
    """
    Stampa i risultati in modo leggibile, includendo un
    commento qualitativo sulla "affidabilità statistica" (dimensione del campione).
    """
    def reliability_comment(n):
        if n >= 30:
            return "alta (n≥30, buona approssimazione alla normale)"
        elif n >= 10:
            return "moderata (10≤n<30, IC ragionevole ma campione contenuto)"
        elif n >= 5:
            return "bassa (5≤n<10, risultati da interpretare con cautela)"
        elif n >= 2:
            return "molto bassa (2≤n<5, stime poco affidabili)"
        elif n == 1:
            return "trascurabile (n=1, non è possibile stimare la variabilità)"
        else:
            return "nessun dato"

    print("=== STATISTICHE GLOBALI ===")
    for metric, stats_dict in results_global.items():
        n = stats_dict["n"]
        print(f"\nMetrica: {metric}")
        print(f"  n                    : {n}")
        print(f"  media                : {stats_dict['mean']:.3f}" if n > 0 else "  media                : n.d.")
        print(f"  deviazione standard  : {stats_dict['std']:.3f}" if n > 1 else "  deviazione standard  : n.d.")
        print(f"  coefficiente variaz. : {stats_dict['cv']:.2f} %" if np.isfinite(stats_dict['cv']) else "  coefficiente variaz. : n.d.")
        print(f"  errore standard      : {stats_dict['se']:.3f}" if n > 1 else "  errore standard      : n.d.")
        if n > 1:
            print(
                f"  IC {int(stats_dict['conf_level']*100)}%         : "
                f"[{stats_dict['ci_low']:.3f}, {stats_dict['ci_high']:.3f}]"
            )
        else:
            print("  IC                    : n.d.")
        print(f"  Affidabilità statistica: {reliability_comment(n)}")

    print("\n\n=== STATISTICHE PER (case, version) ===")
    for (case, version), metric_dict in results_grouped.items():
        print(f"\n--- case={case}, version={version} ---")
        for metric, stats_dict in metric_dict.items():
            n = stats_dict["n"]
            print(f"\n  Metrica: {metric}")
            print(f"    n                    : {n}")
            print(f"    media                : {stats_dict['mean']:.3f}" if n > 0 else "    media                : n.d.")
            print(f"    deviazione standard  : {stats_dict['std']:.3f}" if n > 1 else "    deviazione standard  : n.d.")
            print(f"    coefficiente variaz. : {stats_dict['cv']:.2f} %" if np.isfinite(stats_dict['cv']) else "    coefficiente variaz. : n.d.")
            print(f"    errore standard      : {stats_dict['se']:.3f}" if n > 1 else "    errore standard      : n.d.")
            if n > 1:
                print(
                    f"    IC {int(stats_dict['conf_level']*100)}%         : "
                    f"[{stats_dict['ci_low']:.3f}, {stats_dict['ci_high']:.3f}]"
                )
            else:
                print("    IC                    : n.d.")
            print(f"    Affidabilità statistica: {reliability_comment(n)}")

def reliability_comment(n):
    """
    Restituisce un commento testuale sull'affidabilità statistica
    in base alla dimensione campionaria n.
    """
    if n >= 30:
        return "alta (n≥30, buona approssimazione alla normale)"
    elif n >= 10:
        return "moderata (10≤n<30, IC ragionevole ma campione contenuto)"
    elif n >= 5:
        return "bassa (5≤n<10, risultati da interpretare con cautela)"
    elif n >= 2:
        return "molto bassa (2≤n<5, stime poco affidabili)"
    elif n == 1:
        return "trascurabile (n=1, non è possibile stimare la variabilità)"
    else:
        return "nessun dato"
    
import pandas as pd
import numpy as np

def save_results_to_csv(results_global, results_grouped, output_path):
    """
    Salva i risultati statistici in un file CSV.

    Parametri
    ---------
    results_global : dict
        Output di compute_statistics() per le statistiche globali.
    results_grouped : dict
        Output di compute_statistics() per le statistiche per (case, version).
    output_path : str
        Percorso del file CSV di destinazione.
    """

    rows = []

    # ---- Statistiche globali ----
    for metric, stats in results_global.items():
        n = stats["n"]
        rows.append({
            "scope": "global",
            "case": None,
            "version": None,
            "metric": metric,
            "n": n,
            "mean": stats["mean"],
            "std": stats["std"],
            "cv": stats["cv"],
            "se": stats["se"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "confidence": stats["conf_level"],
            "reliability": reliability_comment(n),
        })

    # ---- Statistiche per gruppi (case, version) ----
    for (case, version), metric_dict in results_grouped.items():
        for metric, stats in metric_dict.items():
            n = stats["n"]
            rows.append({
                "scope": "group",
                "case": case,
                "version": version,
                "metric": metric,
                "n": n,
                "mean": stats["mean"],
                "std": stats["std"],
                "cv": stats["cv"],
                "se": stats["se"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "confidence": stats["conf_level"],
                "reliability": reliability_comment(n),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"[OK] Risultati salvati in: {output_path}")



def plot_statistics(results_global, results_grouped, output_prefix="statistiche"):
    """
    Genera grafici in stile 'da paper' con barre e intervalli di confidenza
    per ciascuna metrica, includendo sia il caso globale che le combinazioni
    (case, version).

    Parametri
    ---------
    results_global : dict
        Statistiche globali come ritornate da compute_statistics().
    results_grouped : dict
        Statistiche per (case, version) come ritornate da compute_statistics().
    output_prefix : str
        Prefisso dei file immagine di output (PNG).
    """

    rows = []

    # ---- Aggiungi righe per il caso GLOBALE ----
    for metric, stats in results_global.items():
        mean = stats["mean"]
        ci_low = stats["ci_low"]
        ci_high = stats["ci_high"]
        n = stats["n"]

        if n is None or n <= 0 or mean is None or np.isnan(mean):
            continue

        if ci_low is not None and ci_high is not None and not (np.isnan(ci_low) or np.isnan(ci_high)):
            err = (ci_high - ci_low) / 2.0
        else:
            err = 0.0

        rows.append({
            "label": "GLOBAL",
            "scope": "global",
            "metric": metric,
            "mean": mean,
            "err": err,
            "n": n,
        })

    # ---- Aggiungi righe per i gruppi (case, version) ----
    for (case, version), metric_dict in results_grouped.items():
        label = f"{case}-{version}"
        for metric, stats in metric_dict.items():
            mean = stats["mean"]
            ci_low = stats["ci_low"]
            ci_high = stats["ci_high"]
            n = stats["n"]

            if n is None or n <= 0 or mean is None or np.isnan(mean):
                continue

            if ci_low is not None and ci_high is not None and not (np.isnan(ci_low) or np.isnan(ci_high)):
                err = (ci_high - ci_low) / 2.0
            else:
                err = 0.0

            rows.append({
                "label": label,
                "scope": "group",
                "metric": metric,
                "mean": mean,
                "err": err,
                "n": n,
            })

    if not rows:
        print("[WARN] Nessun dato valido per i grafici.")
        return

    df_plot = pd.DataFrame(rows)

    # ---- Per ogni metrica, un grafico separato ----
    for metric in df_plot["metric"].unique():
        sub = df_plot[df_plot["metric"] == metric].copy()
        if sub.empty:
            continue

        # Ordina mettendo GLOBAL per primo, poi i gruppi in ordine alfabetico
        sub["is_global"] = sub["scope"].apply(lambda s: 0 if s == "global" else 1)
        sub = sub.sort_values(by=["is_global", "label"])
        sub = sub.reset_index(drop=True)

        labels = sub["label"].tolist()
        means = sub["mean"].to_numpy()
        errs = sub["err"].to_numpy()
        ns = sub["n"].to_numpy()

        x = np.arange(len(sub))

        # ---- Layout più "da paper" ----
        # Figura più larga se ci sono molti gruppi
        width_factor = max(6, len(labels) * 0.5)
        plt.figure(figsize=(width_factor, 4.5))

        # Barre con intervallo di confidenza
        plt.bar(x, means, yerr=errs, capsize=4)
        plt.xticks(x, labels, rotation=45, ha="right")

        plt.ylabel("Media", fontsize=12)
        plt.title(f"{metric}: media con intervallo di confidenza", fontsize=13)

        # Griglia orizzontale leggera (tipica dei paper)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Salva il grafico
        out_path = f"{output_prefix}_{metric}_paper.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] Grafico salvato in: {out_path}")


def plot_validity_edit_distance_pdf(
    csv_path: str,
    output_filename: str = "validity_edit_distance_pdf.png",
    column: str = "validity_edit_distance",
    bins: int = 30,
    title: str = "PDF of validity_edit_distance",
    output_dir: str = "imgs"
):
    """
    Legge un CSV, genera la PDF empirica della colonna validity_edit_distance
    e salva il plot dentro la cartella output_dir.
    """

    # Carica i dati
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"La colonna '{column}' non è presente nel CSV.")

    vals = df[column].dropna()

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Path completo del file di output
    output_path = os.path.join(output_dir, output_filename)

    # Genera il grafico
    plt.figure()
    plt.hist(vals, bins=bins, density=True)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # chiude la figura per evitare overload

    print(f"Plot salvato in: {output_path}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_validity_edit_distance_pdf_kde(
    csv_path: str,
    output_filename: str = "validity_edit_distance_pdf_kde.png",
    column: str = "validity_edit_distance",
    bins: int = 30,
    title: str = "PDF + KDE of validity_edit_distance",
    output_dir: str = "imgs"
):
    """
    Legge un CSV, genera istogramma normalizzato (PDF empirica) + KDE
    per la colonna `validity_edit_distance` e salva il plot in output_dir.

    Parameters
    ----------
    csv_path : str
        Percorso al file CSV.
    output_filename : str, optional
        Nome del file immagine di output.
    column : str, optional
        Colonna da usare per il plot.
    bins : int, optional
        Numero di bin per l'istogramma.
    title : str, optional
        Titolo del grafico.
    output_dir : str, optional
        Cartella in cui salvare l'immagine.
    """

    # Carica il dataset
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise ValueError(f"La colonna '{column}' non è presente nel CSV.")

    # Rimuove NaN
    vals = df[column].dropna().values

    # Crea cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Figura
    plt.figure()

    # Istogramma normalizzato (PDF empirica)
    plt.hist(vals, bins=bins, density=True, alpha=0.5)

    # Calcolo KDE (Gaussian) senza dipendere da scipy
    n = len(vals)
    std = np.std(vals)
    if n < 2 or std == 0:
        raise ValueError("Non ci sono abbastanza dati (o varianza nulla) per calcolare la KDE.")

    # Bandwidth di Silverman
    h = 1.06 * std * n ** (-1/5)

    xs = np.linspace(vals.min(), vals.max(), 500)
    ys = np.zeros_like(xs)
    for v in vals:
        ys += np.exp(-0.5 * ((xs - v) / h) ** 2) / (np.sqrt(2 * np.pi) * h)
    ys /= n

    # Linea KDE
    plt.plot(xs, ys)

    # Etichette e titolo
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()

    # Salvataggio e chiusura
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot salvato in: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calcolo statistiche per graph_edit_distance e validity_edit_distance.")
    parser.add_argument("csv_path", help="Percorso del file CSV con i dati.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Livello di confidenza per gli intervalli (default 0.95).",
    )

    args = parser.parse_args()

    results_global, results_grouped = compute_statistics(args.csv_path, confidence=args.confidence)
    print_results(results_global, results_grouped)
    
    # Salva risultati in CSV
    save_results_to_csv(results_global, results_grouped, "results/statistiche_edit_distance.csv")

    # Genera grafici
    plot_statistics(results_global, results_grouped, output_prefix="imgs/statistiche_edit_distance")
    #plot_validity_edit_distance_pdf(args.csv_path)
    plot_validity_edit_distance_pdf_kde(args.csv_path)
