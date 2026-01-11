#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_anova(df, metric):
    """
    ANOVA a due fattori (case, version) con interazione per una data metrica.
    Ritorna: (df_sub, modello, tabella_anova)
    """
    sub = df.dropna(subset=[metric, "case", "version"])
    model = ols(f"{metric} ~ C(case) * C(version)", data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return sub, model, anova_table


def compute_effect_sizes(anova_table):
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


def marginal_means(df, metric):
    """
    Effetti marginali: medie della metrica per livello di case e version.
    Ritorna (df_case, df_version).
    """
    sub = df.dropna(subset=[metric, "case", "version"])
    mm_case = (
        sub.groupby("case")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
    )
    mm_version = (
        sub.groupby("version")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
    )
    return mm_case, mm_version


def tukey_to_dataframe(tukey_result):
    """
    Converte un risultato Tukey HSD (statsmodels) in un DataFrame pandas.
    """
    data = tukey_result._results_table.data
    df_tuk = pd.DataFrame(data[1:], columns=data[0])
    return df_tuk


def run_posthoc(df, metric, factor):
    """
    Esegue test post-hoc Tukey HSD per un fattore (case o version).
    Ritorna il DataFrame con i risultati.
    """
    sub = df.dropna(subset=[metric, factor])
    tukey = pairwise_tukeyhsd(endog=sub[metric], groups=sub[factor])
    df_tuk = tukey_to_dataframe(tukey)
    return df_tuk


def doe_for_metric(df, metric, prefix_out="DOE"):
    """
    Esegue l'intera pipeline DOE per una singola metrica:
      - ANOVA
      - effect sizes (η², ω²)
      - marginal means
      - post-hoc Tukey per case e version
    Salva i risultati in vari CSV e ritorna un dict con i principali oggetti.
    """
    print("\n" + "=" * 80)
    print(f"DOE per metrica: {metric}")
    print("=" * 80)

    # ANOVA
    sub, model, anova_table = run_anova(df, metric)
    print("\n[ANOVA]")
    print(anova_table)

    # Effect sizes (eta squared, omega squared)
    eff_sizes = compute_effect_sizes(anova_table)
    print("\n[Effect sizes (eta², ω²)]")
    print(eff_sizes)

    # Effetti marginali
    mm_case, mm_version = marginal_means(df, metric)
    print("\n[Effetti marginali - case]")
    print(mm_case)
    print("\n[Effetti marginali - version]")
    print(mm_version)

    # Post-hoc Tukey
    tukey_case = run_posthoc(df, metric, "case")
    tukey_version = run_posthoc(df, metric, "version")
    print("\n[Post-hoc Tukey HSD - case]")
    print(tukey_case)
    print("\n[Post-hoc Tukey HSD - version]")
    print(tukey_version)

    # Salvataggio CSV
    anova_table.to_csv(f"{prefix_out}_{metric}_anova.csv")
    eff_sizes.to_csv(f"{prefix_out}_{metric}_effect_sizes.csv", index=False)
    mm_case.to_csv(f"{prefix_out}_{metric}_marginal_means_case.csv", index=False)
    mm_version.to_csv(f"{prefix_out}_{metric}_marginal_means_version.csv", index=False)
    tukey_case.to_csv(f"{prefix_out}_{metric}_tukey_case.csv", index=False)
    tukey_version.to_csv(f"{prefix_out}_{metric}_tukey_version.csv", index=False)

    print("\n[OK] File CSV salvati per la metrica:", metric)

    return {
        "data": sub,
        "model": model,
        "anova": anova_table,
        "effect_sizes": eff_sizes,
        "mm_case": mm_case,
        "mm_version": mm_version,
        "tukey_case": tukey_case,
        "tukey_version": tukey_version,
    }

import matplotlib.pyplot as plt
import numpy as np

def plot_main_effects(df, metric, output_prefix="DOE_main"):
    """
    Genera i grafici dei main effects (case e version) per una data metrica,
    in stile paper.

    - Asse X: livelli del fattore (case / version)
    - Asse Y: media della metrica (marginal mean)
    - Barre di errore: IC 95% ~ mean ± 1.96 * SE

    Parametri
    ---------
    df : pandas.DataFrame
        Dataset di input (contiene colonne: metric, case, version).
    metric : str
        Nome della metrica (es. 'graph_edit_distance').
    output_prefix : str
        Prefisso per il nome dei file PNG generati.
    """
    sub = df.dropna(subset=[metric, "case", "version"]).copy()
    if sub.empty:
        print(f"[WARN] Nessun dato valido per la metrica {metric} nei grafici dei main effects.")
        return

    z = 1.96  # per IC 95% approssimato

    # ===== MAIN EFFECT: CASE =====
    g_case = (
        sub.groupby("case")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g_case["se"] = g_case["std"] / np.sqrt(g_case["count"])
    g_case["err"] = z * g_case["se"]

    # Ordiniamo i livelli di case alfabeticamente per coerenza visiva
    g_case = g_case.sort_values("case")
    x_case = np.arange(len(g_case))

    plt.figure(figsize=(max(6, len(g_case) * 0.7), 4.0))
    plt.errorbar(
        x_case,
        g_case["mean"],
        yerr=g_case["err"],
        fmt="-o",
        capsize=4,
    )
    plt.xticks(x_case, g_case["case"], rotation=45, ha="right")
    plt.ylabel("Media", fontsize=12)
    plt.title(f"Main effect di case su {metric}", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path_case = f"{output_prefix}_{metric}_main_case.png"
    plt.savefig(out_path_case, dpi=300)
    plt.close()
    print(f"[OK] Grafico main effect (case) salvato in: {out_path_case}")

    # ===== MAIN EFFECT: VERSION =====
    g_ver = (
        sub.groupby("version")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g_ver["se"] = g_ver["std"] / np.sqrt(g_ver["count"])
    g_ver["err"] = z * g_ver["se"]

    # Ordiniamo le versioni in modo naturale (es. v1, v2, v3 ...)
    # Se sono stringhe tipo 'v1', 'v2', ecc., proviamo a ordinare per numero interno
    try:
        sort_key = g_ver["version"].str.extract(r"(\d+)").astype(float)[0]
        g_ver = g_ver.iloc[np.argsort(sort_key.values)]
    except Exception:
        g_ver = g_ver.sort_values("version")

    x_ver = np.arange(len(g_ver))

    plt.figure(figsize=(max(5, len(g_ver) * 0.7), 4.0))
    plt.errorbar(
        x_ver,
        g_ver["mean"],
        yerr=g_ver["err"],
        fmt="-o",
        capsize=4,
    )
    plt.xticks(x_ver, g_ver["version"], rotation=0, ha="center")
    plt.ylabel("Media", fontsize=12)
    plt.title(f"Main effect di version su {metric}", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path_ver = f"{output_prefix}_{metric}_main_version.png"
    plt.savefig(out_path_ver, dpi=300)
    plt.close()
    print(f"[OK] Grafico main effect (version) salvato in: {out_path_ver}")

def run_doe_on_csv(csv_path):
    """
    Funzione principale:
      - legge il CSV
      - esegue il DOE per entrambe le metriche:
          * graph_edit_distance
          * validity_edit_distance
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
        description="DOE 2-way (case, version) su distanze di edit."
    )
    parser.add_argument(
        "csv_path", help="Percorso del file CSV (es. dati_validity.csv)"
    )
    args = parser.parse_args()

    run_doe_on_csv(args.csv_path)