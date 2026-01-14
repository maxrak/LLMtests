#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script per Design of Experiment (DoE) su dati CSV.

Funzioni principali:
1) Legge un file di configurazione JSON con:
   - nomi dei fattori
   - nomi degli indici (metriche di risultato)
   - per ogni indice: se "better" è "high" o "low"
   - livello di confidenza (es. 0.95)
   - nome del file CSV dei dati

2) Legge i dati dal CSV.

3) Per gli indici binari (0/1):
   - esegue regressione logistica con tutti i fattori
   - stampa tabella con info generali del modello
   - stampa tabella dei coefficienti
   - calcola Likelihood Ratio test (LR) per:
        * ogni singolo fattore
        * ogni coppia di fattori (combinazioni di fattori)

4) Per gli indici continui:
   - esegue ANOVA (modello lineare) con tutti i fattori
   - esegue confrontationi multipli (Tukey HSD) per i fattori categorici con >2 livelli
   - produce una tabella di significatività di fattori e combinazioni (F-test via modelli annidati)

5) Crea una tabella descrittiva per tutte le combinazioni dei fattori:
   - conteggio, media (o proporzione), stima di significatività per ciascuna combinazione
     rispetto al valore medio globale (binomiale per indici 0/1, t-test per continui)

6) Tabella riassuntiva conclusiva:
   - per ogni indice mostra i fattori (e combinazioni) più significativi (p-value più bassi).
"""

import sys
import json
import itertools
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pd.set_option('display.max_rows', None) # Mostra tutte le righe
pd.set_option('display.max_columns', None) # Mostra tutte le colonne
pd.set_option('display.width', 2000)       # Imposta la larghezza (ad esempio, 1000 caratteri)

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def is_binary_series(s: pd.Series) -> bool:
    """Ritorna True se la serie contiene solo 0/1 (ignorando NaN)."""
    vals = pd.unique(s.dropna())
    return len(vals) > 1 and set(vals).issubset({0, 1})


def factor_term(col: str, series: pd.Series) -> str:
    """
    Costruisce il termine di formula per statsmodels:
    - C(col) se categorico/oggetto
    - col se numerico
    """
    if series.dtype == "object" or str(series.dtype).startswith("category") or series.dtype == "bool":
        return f"C({col})"
    else:
        return col


def build_formula(response: str, df: pd.DataFrame, factors: List[str]) -> str:
    """Costruisce formula di regressione 'response ~ fattori'."""
    terms = [factor_term(f, df[f]) for f in factors]
    rhs = " + ".join(terms) if terms else "1"
    return f"{response} ~ {rhs}"


def get_conf_level_alpha(conf_level: float) -> float:
    """Restituisce alpha (1 - conf_level)."""
    return 1.0 - conf_level


# -------------------------------------------------------------------
# 3) Analisi logistica per indici binari
# -------------------------------------------------------------------

def logistic_analysis(df: pd.DataFrame,
                      factors: List[str],
                      index: str,
                      conf_level: float,
                      max_comb_order: int = 2) -> Dict[str, Any]:
    """
    Esegue regressione logistica per un indice binario:
    - modello completo con tutti i fattori
    - coefficenti
    - LR test per singoli fattori e combinazioni di fattori fino a 'max_comb_order'.
    """
    formula_full = build_formula(index, df, factors)
    print(f"\n=== LOGISTIC REGRESSION per indice '{index}' ===")
    print(f"Formula: {formula_full}")

    # Modello completo
    model_full = smf.logit(formula_full, data=df).fit(disp=False)

    # Modello nullo (solo intercetta) per pseudo-R2
    model_null = smf.logit(f"{index} ~ 1", data=df).fit(disp=False)

    # Info modello
    model_info = {
        "n_obs": int(model_full.nobs),
        "log_likelihood": float(model_full.llf),
        "log_likelihood_null": float(model_null.llf),
        "pseudo_r2_mcfadden": 1 - (model_full.llf / model_null.llf),
        "aic": float(model_full.aic),
        "bic": float(model_full.bic),
    }
    print("\n--- Info generali modello ---")
    print(pd.Series(model_info).to_frame("value"))

    # Tabella coefficienti
    coef_table = model_full.summary2().tables[1]
    print("\n--- Coefficienti (logit) ---")
    print(coef_table)

    # LR tests per fattori e combinazioni
    lr_rows = []
    all_factor_subsets = []
    for k in range(1, max_comb_order + 1):
        for subset in itertools.combinations(factors, k):
            all_factor_subsets.append(subset)

    for subset in all_factor_subsets:
        # fattori rimossi nel modello ridotto
        remaining = [f for f in factors if f not in subset]
        formula_reduced = build_formula(index, df, remaining)
        model_reduced = smf.logit(formula_reduced, data=df).fit(disp=False)

        lr_stat = 2 * (model_full.llf - model_reduced.llf)
        df_diff = model_full.df_model - model_reduced.df_model
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

        lr_rows.append({
            "factors_removed": " + ".join(subset),
            "df_diff": df_diff,
            "lr_stat": lr_stat,
            "p_value": p_value,
            "significant": p_value < get_conf_level_alpha(conf_level)
        })

    lr_df = pd.DataFrame(lr_rows).sort_values("p_value")
    print("\n--- Likelihood Ratio tests (fattori / combinazioni) ---")
    print(lr_df)

    return {
        "model_info": model_info,
        "coef_table": coef_table,
        "lr_tests": lr_df,
        "model_full": model_full,
    }


# -------------------------------------------------------------------
# 4) ANOVA + Tukey per indici continui
# -------------------------------------------------------------------

def anova_analysis(df: pd.DataFrame,
                   factors: List[str],
                   index: str,
                   conf_level: float,
                   max_comb_order: int = 2) -> Dict[str, Any]:
    """
    Esegue:
    - Modello lineare (OLS)
    - ANOVA tipo II
    - Confronti multipli Tukey HSD per fattori categorici con >2 livelli
    - F-test per combinazioni di fattori (modelli annidati).
    """
    formula_full = build_formula(index, df, factors)
    print(f"\n=== ANOVA / OLS per indice '{index}' ===")
    print(f"Formula: {formula_full}")

    model_full = smf.ols(formula_full, data=df).fit()

    print("\n--- Riassunto modello OLS ---")
    print(model_full.summary())

    # ANOVA tipo II
    anova_df = anova_lm(model_full, typ=2)
    anova_df["significant"] = anova_df["PR(>F)"] < get_conf_level_alpha(conf_level)
    print("\n--- ANOVA (Type II) ---")
    print(anova_df)

    # Tukey HSD per fattori categorici con >2 livelli
    tukey_results = {}
    alpha = get_conf_level_alpha(conf_level)

    for f in factors:
        s = df[f]
        is_cat = s.dtype == "object" or str(s.dtype).startswith("category") or s.dtype == "bool"
        if is_cat and s.nunique() > 2:
            print(f"\n--- Tukey HSD per fattore '{f}' ---")
            tukey = pairwise_tukeyhsd(endog=df[index], groups=df[f], alpha=alpha)
            print(tukey.summary())
            tukey_results[f] = tukey

    # F-test per combinazioni di fattori (modelli annidati)
    comb_rows = []
    all_factor_subsets = []
    for k in range(1, max_comb_order + 1):
        for subset in itertools.combinations(factors, k):
            all_factor_subsets.append(subset)

    for subset in all_factor_subsets:
        remaining = [f for f in factors if f not in subset]
        formula_reduced = build_formula(index, df, remaining)
        model_reduced = smf.ols(formula_reduced, data=df).fit()

        # model_full deve includere model_reduced (modelli annidati)
        F, p_value, df_diff = model_full.compare_f_test(model_reduced)
        comb_rows.append({
            "factors_removed": " + ".join(subset),
            "df_diff": df_diff,
            "F_stat": F,
            "p_value": p_value,
            "significant": p_value < alpha
        })

    comb_df = pd.DataFrame(comb_rows).sort_values("p_value")
    print("\n--- F-test per combinazioni di fattori (modelli annidati) ---")
    print(comb_df)

    return {
        "anova_table": anova_df,
        "tukey_results": tukey_results,
        "combination_tests": comb_df,
        "model_full": model_full
    }


# -------------------------------------------------------------------
# 5) Tabella descrittiva per configurazioni fattori
# -------------------------------------------------------------------
def descriptive_by_configuration(df: pd.DataFrame,
                                 factors: List[str],
                                 index: str,
                                 conf_level: float,
                                 better: str,
                                 min_n_for_test: int = 2,
                                 std_tol: float = 1e-8) -> pd.DataFrame:
    """
    Per ogni combinazione di fattori, calcola:
    - n
    - mean_value
    - test di significatività vs media globale:
        * binomiale se indice binario
        * t-test se continuo, n >= min_n_for_test e varianza non ~0
    - flag:
        * n_too_small_for_test
        * low_variance_group
    """
    alpha = get_conf_level_alpha(conf_level)
    s = df[index].dropna()

    is_binary = is_binary_series(s)
    global_mean = s.mean()

    rows = []
    grouped = df.groupby(factors, dropna=False)

    for combo, sub in grouped:
        if not isinstance(combo, tuple):
            combo = (combo,)

        y = sub[index].dropna()
        n = len(y)
        if n == 0:
            continue

        mean_val = y.mean()
        n_too_small = n < min_n_for_test
        low_variance_group = False

        # ------------------- calcolo p-value -------------------
        if is_binary:
            # binomiale sempre definito (ma n_too_small rimane un flag informativo)
            successes = y.sum()
            res = stats.binomtest(successes, n, p=global_mean, alternative='two-sided')
            p_value = res.pvalue

        else:
            # continuo
            if n_too_small:
                # niente test se ho troppi pochi punti
                p_value = np.nan
            else:
                # controllo varianza: se std ~ 0 evito il t-test
                std_y = y.std(ddof=1)
                if not np.isfinite(std_y) or abs(std_y) < std_tol:
                    p_value = np.nan
                    low_variance_group = True
                else:
                    t_stat, p_value = stats.ttest_1samp(y, popmean=global_mean)

        if np.isfinite(p_value):
            significant = p_value < alpha
        else:
            significant = False

        # direzione rispetto alla media globale
        direction = "not_significant"
        if significant:
            if mean_val > global_mean:
                direction = "higher"
            elif mean_val < global_mean:
                direction = "lower"

        # direzione rispetto al criterio "better"
        better_flag = None
        if direction in ("higher", "lower"):
            if better == "high":
                better_flag = "better" if direction == "higher" else "worse"
            elif better == "low":
                better_flag = "better" if direction == "lower" else "worse"

        row = {
            **{f: v for f, v in zip(factors, combo)},
            "n": n,
            "mean_value": mean_val,
            "is_binary_index": is_binary,
            "global_mean": global_mean,
            "p_value": p_value,
            "significant": significant,
            "direction_raw": direction,
            "direction_vs_better": better_flag,
            # 👇 flag che già avevamo
            "n_too_small_for_test": n_too_small,
            # 👇 nuovo flag per i warning di “precision loss”
            "low_variance_group": low_variance_group,
        }
        rows.append(row)

    desc_df = pd.DataFrame(rows)
    return desc_df

# -------------------------------------------------------------------
# 6) Tabella riassuntiva conclusiva
# -------------------------------------------------------------------

def build_summary_table(all_results: Dict[str, Dict[str, Any]],
                        conf_level: float) -> pd.DataFrame:
    """
    all_results: dizionario per indice:
        {
          index_name: {
             "type": "binary" / "continuous",
             "lr_tests": <df> (per binary),
             "anova_table": <df> (per continuous),
             "combination_tests": <df> (per continuous),
          }
        }

    Ritorna una tabella riassuntiva con i fattori/combinazioni più significativi per ogni indice.
    """
    alpha = get_conf_level_alpha(conf_level)
    rows = []

    for idx_name, res in all_results.items():
        idx_type = res["type"]

        if idx_type == "binary":
            # prendo LR test
            lr_df = res["lr_tests"]
            for _, r in lr_df.iterrows():
                if pd.isna(r["p_value"]):
                    continue
                rows.append({
                    "index": idx_name,
                    "index_type": idx_type,
                    "test_type": "LR",
                    "factors_or_combination": r["factors_removed"],
                    "p_value": r["p_value"],
                    "significant": r["p_value"] < alpha
                })

        elif idx_type == "continuous":
            # ANOVA main effects
            anova_df = res["anova_table"]
            for factor_name, r in anova_df.iterrows():
                if factor_name == "Residual":
                    continue
                p_val = r["PR(>F)"]
                rows.append({
                    "index": idx_name,
                    "index_type": idx_type,
                    "test_type": "ANOVA_main",
                    "factors_or_combination": factor_name,
                    "p_value": p_val,
                    "significant": p_val < alpha
                })

            # combinazioni
            comb_df = res["combination_tests"]
            for _, r in comb_df.iterrows():
                rows.append({
                    "index": idx_name,
                    "index_type": idx_type,
                    "test_type": "ANOVA_combination",
                    "factors_or_combination": r["factors_removed"],
                    "p_value": r["p_value"],
                    "significant": r["p_value"] < alpha
                })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["index", "p_value"])
    return summary_df


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main(config_path: str):
    # 1) Legge file di configurazione
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    csv_file = config["csv_file"]
    sep = config.get("separator", ",")
    factors: List[str] = config["factors"]
    indices_cfg: Dict[str, Dict[str, Any]] = config["indices"]
    conf_level: float = config.get("confidence_level", 0.95)
    max_comb_order: int = config.get("max_combination_order", 2)
    min_n_for_test: int = config.get("min_n_for_test", 2)

    print("===========================================")
    print(f"Config file: {config_path}")
    print(f"CSV file:    {csv_file}")
    print(f"Separator:   '{sep}'")
    print(f"Fattori:     {factors}")
    print(f"Indici:      {list(indices_cfg.keys())}")
    print(f"Conf level:  {conf_level}")
    print("===========================================")

    # 2) Legge CSV
    df = pd.read_csv(csv_file, sep=sep)

    # Filtra solo colonne necessarie
    needed_cols = set(factors) | set(indices_cfg.keys())
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    df = df[list(needed_cols)].copy()

    all_results_for_summary = {}
    all_desc_tables = {}

    # Analisi per ciascun indice
    for idx_name, idx_info in indices_cfg.items():
        better = idx_info.get("better", "high").lower()
        s = df[idx_name]
        if is_binary_series(s):
            idx_type = "binary"
            res_log = logistic_analysis(df, factors, idx_name, conf_level,
                                        max_comb_order=max_comb_order)

            all_results_for_summary[idx_name] = {
                "type": "binary",
                "lr_tests": res_log["lr_tests"],
            }

        else:
            idx_type = "continuous"
            res_anova = anova_analysis(df, factors, idx_name, conf_level,
                                       max_comb_order=max_comb_order)

            all_results_for_summary[idx_name] = {
                "type": "continuous",
                "anova_table": res_anova["anova_table"],
                "combination_tests": res_anova["combination_tests"],
            }

        # 5) tabella descrittiva per config fattori
        desc_df = descriptive_by_configuration(
            df, factors, idx_name, conf_level, better, min_n_for_test=min_n_for_test
        )
        print(f"\n--- Tabella descrittiva per indice '{idx_name}' ---")
        print(desc_df)
        all_desc_tables[idx_name] = desc_df

    # 6) tabella riassuntiva
    summary_df = build_summary_table(all_results_for_summary, conf_level)
    print("\n===========================================")
    print("TABElLA RIASSUNTIVA FATTORI PIÙ SIGNIFICATIVI PER OGNI INDICE")
    print("===========================================")
    print(summary_df)

    # opzionale: salvare le tabelle in CSV
    out_prefix = config.get("output_prefix", "doe_output")

    summary_df.to_csv(f"{out_prefix}_summary.csv", index=False)
    for idx_name, desc_df in all_desc_tables.items():
        desc_df.to_csv(f"{out_prefix}_desc_{idx_name}.csv", index=False)

    print("\nFile di output salvati come:")
    print(f"- {out_prefix}_summary.csv")
    for idx_name in indices_cfg.keys():
        print(f"- {out_prefix}_desc_{idx_name}.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python doe_analysis.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])