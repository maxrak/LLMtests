#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import os


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf, edgeitems=1000)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Fattori del DOE
FACTORS = ["case", "mode", "version", "RAG"]


def run_anova(df: pd.DataFrame, metric: str):
    """
    ANOVA multifattoriale solo con main effects
    (case, mode, version, RAG).

    NOTA: con i tuoi dati reali alcune combinazioni case–version–RAG
    non esistono; per questo motivo il modello con tutte le interazioni
    dava problemi di rango. Qui usiamo un modello più parsimonioso:

        metric ~ C(case) + C(version) + C(RAG)
        (+ C(mode) se ci fossero più livelli)

    Ritorna: (df_sub, modello, tabella_anova)
    """
    cols = [metric] + FACTORS
    sub = df.dropna(subset=cols).copy()

    # Consideriamo solo i fattori che hanno più di un livello nel subset
    active_factors = [f for f in FACTORS if sub[f].nunique() > 1]

    # SOLO MAIN EFFECTS (niente "*", solo "+")
    rhs = " + ".join(f"C({f})" for f in active_factors)
    formula = f"{metric} ~ {rhs}"

    model = ols(formula, data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    return sub, model, anova_table

def compute_effect_sizes(anova_table: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola η² (eta squared) e ω² (omega squared) per ogni effetto dell'ANOVA.
    Ritorna un DataFrame con colonne:
    - effect
    - eta_sq, eta_sq_pct
    - omega_sq, omega_sq_pct
    """
    ss_total = anova_table["sum_sq"].sum()
    ss_res = anova_table.loc["Residual", "sum_sq"]
    df_res = anova_table.loc["Residual", "df"]
    ms_res = ss_res / df_res

    rows = []
    for term in anova_table.index:
        if term == "Residual":
            continue
        ss = anova_table.loc[term, "sum_sq"]
        df = anova_table.loc[term, "df"]
        eta2 = ss / ss_total
        omega2 = (ss - df * ms_res) / (ss_total + ms_res)
        rows.append(
            {
                "effect": term,
                "eta_sq": eta2,
                "eta_sq_pct": eta2 * 100.0,
                "omega_sq": omega2,
                "omega_sq_pct": omega2 * 100.0,
            }
        )
    return pd.DataFrame(rows)


def marginal_means(df: pd.DataFrame, metric: str):
    """
    Effetti marginali: medie della metrica per livello di ogni fattore.
    Ritorna un dict {fattore: DataFrame}.
    """
    cols = [metric] + FACTORS
    sub = df.dropna(subset=cols).copy()
    mm = {}
    for fac in FACTORS:
        g = (
            sub.groupby(fac)[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "mean"})
        )
        mm[fac] = g
    return mm


def tukey_to_dataframe(tukey_result) -> pd.DataFrame:
    """
    Converte un risultato Tukey HSD (statsmodels) in un DataFrame pandas.
    """
    data = tukey_result._results_table.data
    df_tuk = pd.DataFrame(data[1:], columns=data[0])
    return df_tuk


def run_posthoc(df: pd.DataFrame, metric: str, factor: str) -> pd.DataFrame:
    """
    Esegue test post-hoc Tukey HSD per un fattore (case, mode, version o RAG).
    Ritorna il DataFrame con i risultati.
    """
    sub = df.dropna(subset=[metric, factor]).copy()
    tukey = pairwise_tukeyhsd(endog=sub[metric], groups=sub[factor])
    df_tuk = tukey_to_dataframe(tukey)
    return df_tuk


def doe_for_metric(df: pd.DataFrame, metric: str, prefix_out: str = "DOE"):
    """
    Esegue l'intera pipeline DOE per una singola metrica:
      - ANOVA
      - effect sizes (η², ω²)
      - marginal means per tutti i fattori
      - post-hoc Tukey per tutti i fattori

    Salva i risultati in vari CSV con prefisso:
      {prefix_out}_{metric}_...
    """
    sub, model, anova_table = run_anova(df, metric)
    print(f"\n================= METRICA: {metric} =================")
    print("\n[ANOVA]")
    print(anova_table)

    # Effect sizes (eta squared, omega squared)
    eff_sizes = compute_effect_sizes(anova_table)
    print("\n[Effect sizes (eta², ω²)]")
    print(eff_sizes)

    # Effetti marginali
    mm_dict = marginal_means(df, metric)
    for fac, mm_df in mm_dict.items():
        print(f"\n[Effetti marginali - {fac}]")
        print(mm_df)

    # Post-hoc Tukey per ciascun fattore
    tukey_dict = {}
    for fac in FACTORS:
        try:
            tuk = run_posthoc(df, metric, fac)
        except Exception as e:
            print(f"\n[WARN] Impossibile eseguire Tukey per {fac} sulla metrica {metric}: {e}")
            tuk = None
        tukey_dict[fac] = tuk
        if tuk is not None:
            print(f"\n[Post-hoc Tukey HSD - {fac}]")
            print(tuk)

    # Salvataggio CSV
    # ANOVA ed effect sizes
    path_anova = f"{prefix_out}_{metric}_anova.csv"
    _ensure_dir(path_anova)
    anova_table.to_csv(path_anova)

    path_eff = f"{prefix_out}_{metric}_effect_sizes.csv"
    _ensure_dir(path_eff)
    eff_sizes.to_csv(path_eff, index=False)

    # Marginal means
    for fac, mm_df in mm_dict.items():
        path_mm = f"{prefix_out}_{metric}_marginal_means_{fac}.csv"
        _ensure_dir(path_mm)
        mm_df.to_csv(path_mm, index=False)

    # Tukey
    for fac, tuk in tukey_dict.items():
        if tuk is not None:
            path_tk = f"{prefix_out}_{metric}_tukey_{fac}.csv"
            _ensure_dir(path_tk)
            tuk.to_csv(path_tk, index=False)

    print("\n[OK] File CSV salvati per la metrica:", metric)

    return {
        "data": sub,
        "model": model,
        "anova": anova_table,
        "effect_sizes": eff_sizes,
        "marginal_means": mm_dict,
        "tukey": tukey_dict,
    }


def plot_main_effects(df: pd.DataFrame, metric: str, output_prefix: str = "DOE_main"):
    """
    Genera i grafici dei main effects (case, mode, version, RAG) per una data metrica.

    - Asse X: livelli del fattore
    - Asse Y: media della metrica
    - Barre di errore: IC 95% ~ mean ± 1.96 * SE
    """
    cols = [metric] + FACTORS
    sub = df.dropna(subset=cols).copy()

    if sub.empty:
        print(f"[WARN] Nessun dato valido per la metrica {metric} nei grafici dei main effects.")
        return

    z = 1.96  # IC 95%

    for fac in FACTORS:
        g = (
            sub.groupby(fac)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        if g.empty:
            print(f"[WARN] Nessun dato per il fattore {fac} (metrica {metric}).")
            continue

        g["se"] = g["std"] / np.sqrt(g["count"])
        g["err"] = z * g["se"]

        # Ordine "naturale" per i livelli (se ci sono numeri all'interno, tipo v1, v2, ...)
        try:
            sort_key = g[fac].astype(str).str.extract(r"(\d+)").astype(float)[0]
            g = g.iloc[np.argsort(sort_key.values)]
        except Exception:
            g = g.sort_values(fac)

        x = np.arange(len(g))

        plt.figure(figsize=(7, 4))
        plt.bar(
            x,
            g["mean"],
            yerr=g["err"],
            align="center",
            alpha=0.8,
            ecolor="black",
            capsize=4,
        )
        plt.xticks(x, g[fac].astype(str), rotation=45, ha="right")
        plt.ylabel("Media")
        plt.title(f"Main effect di {fac} su {metric}")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = f"{output_prefix}_{metric}_main_{fac}.png"
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] Grafico main effect ({fac}) salvato in: {out_path}")


def run_doe_on_csv(csv_path: str):
    """
    Funzione principale:
      - legge il CSV (es. dati_validity.csv)
      - esegue il DOE per entrambe le metriche:
          * graph_edit_distance
          * validity_edit_distance
      - genera i grafici dei main effects
    """
    df = pd.read_csv(csv_path)

    metrics = ["graph_edit_distance", "validity_edit_distance"]

    results = {}
    for metric in metrics:
        res = doe_for_metric(df, metric, prefix_out="validity/DOE")
        results[metric] = res
        # grafici dei main effects
        plot_main_effects(df, metric, output_prefix="imgs/DOE")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "DOE multifattoriale (case, mode, version, RAG) "
            "su distanze di edit."
        )
    )
    parser.add_argument(
        "csv_path", help="Percorso del file CSV (es. dati_validity.csv)"
    )
    args = parser.parse_args()

    run_doe_on_csv(args.csv_path)