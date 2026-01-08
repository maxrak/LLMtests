import csv
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAZIONE DI DEFAULT
# ==========================

# Nome predefinito del CSV per l'analisi
DEFAULT_CSV_PATH = "dati_processo.csv"

# Nomi colonna nel CSV (compatibili con l'analisi)
COL_CASE = "case"
COL_VERSION = "version"
COL_MODE = "mode"
COL_SUCCESS = "success"
COL_ITER = "iterations"

GROUP_COLS = [COL_CASE, COL_VERSION, COL_MODE]

# Parametri statistici di default
ALPHA_DEFAULT = 0.05
H_P_DEFAULT = 0.05     # ampiezza target ± per p (probabilità di successo)
H_ITER_DEFAULT = 2.0   # ampiezza target ± per la media iterazioni (successi)


import csv

# campi richiesti per l'analisi
COL_CASE = "case"
COL_VERSION = "version"
COL_MODE = "mode"
COL_SUCCESS = "success"
COL_ITER = "iterations"

CSV_FIELDNAMES = [COL_CASE, COL_VERSION, COL_MODE, COL_SUCCESS, COL_ITER]


def CSVgen(rows, output_csv_path="dati_processo.csv"):
    """
    Genera un CSV per l'analisi a partire da 'rows'.

    Mappatura:
      case       -> rec["apps"]["id"]
      version    -> rec["version"]
      iterations -> rec["iterations"]
      mode       -> rec["summarized"]:
                       0 -> "notsummarized"
                       1 -> "summarized"
      success    -> 1 se iterations < 30, altrimenti 0
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for rec in rows:
            # case, version, iterations
            case = rec.get("apps", {}).get("id", None)
            version = rec.get("version", None)
            iterations = rec.get("iterations", None)

            # --- mode da booleano 0/1 (campo 'summarized') ---
            raw_mode = rec.get("summarized", 0)

            # gestiamo int, bool, stringhe "0"/"1"
            try:
                # True -> 1, False -> 0, "0"/"1" -> 0/1, ecc.
                raw_val = int(raw_mode)
            except (TypeError, ValueError):
                # fallback: consideriamo non summarized
                raw_val = 0

            if raw_val == 1:
                mode = "summarized"
            else:
                mode = "notsummarized"

            # --- successo/fallimento in base alle iterations ---
            try:
                iters_int = int(iterations)
            except Exception:
                # se non convertibile, trattiamo come fallimento
                iters_int = 30

            success = 1 if iters_int < 30 else 0

            writer.writerow({
                COL_CASE: case,
                COL_VERSION: version,
                COL_MODE: mode,
                COL_SUCCESS: success,
                COL_ITER: iterations,
            })

    return output_csv_path


# ==========================
# FUNZIONI DI SUPPORTO STIMA
# ==========================

def parse_success_column(s):
    """
    Converte la colonna di successo in booleano (True = successo, False = fallimento).
    Si aspetta 0/1, True/False, o stringhe tipo "success"/"fail".
    """
    if isinstance(s, str):
        s_lower = s.strip().lower()
        if s_lower in ["success", "ok", "true", "1", "yes"]:
            return True
        elif s_lower in ["fail", "failure", "ko", "false", "0", "no"]:
            return False
    if isinstance(s, (int, float, np.integer, np.floating)):
        return bool(s)
    return False


def wilson_ci(k, n, alpha=0.05):
    """
    Intervallo di confidenza di Wilson per una proporzione.

    Parameters
    ----------
    k : int
        Numero di successi.
    n : int
        Numero totale di prove.

    Returns
    -------
    (p_hat, lower, upper)
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)

    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / (2 * n)
    centre = p_hat + z ** 2 / (2 * n)
    margin = z * sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return p_hat, lower, upper


def required_n_for_proportion(p_hat, h, alpha=0.05):
    """
    Numero minimo di campioni per avere intervallo di confidenza ±h per una proporzione.
    Usa p_hat se disponibile, altrimenti p=0.5 (caso peggiore).
    """
    if np.isnan(p_hat):
        p = 0.5
    else:
        p = p_hat
    z = norm.ppf(1 - alpha / 2)
    n_req = (z ** 2 * p * (1 - p)) / (h ** 2)
    return n_req


def mean_ci_normal(x, alpha=0.05):
    """
    Intervallo di confidenza per la media assumendo (approssimativamente) normalità
    o CLT (n sufficientemente grande).
    """
    x = np.array(x)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    m = x.mean()
    s = x.std(ddof=1) if n > 1 else 0.0
    z = norm.ppf(1 - alpha / 2)

    if n > 1:
        half_width = z * s / sqrt(n)
        lower = m - half_width
        upper = m + half_width
    else:
        half_width = np.nan
        lower = np.nan
        upper = np.nan

    return m, s, lower, upper, n


def required_n_for_mean(s, h, alpha=0.05):
    """
    Numero minimo di campioni per stimare la media con intervallo ±h,
    data una stima s della deviazione standard.
    """
    if s is None or np.isnan(s) or s == 0:
        return np.nan
    z = norm.ppf(1 - alpha / 2)
    n_req = (z * s / h) ** 2
    return n_req


def lr_test(full_model, reduced_model):
    """
    Likelihood Ratio test tra:
      - full_model: modello completo
      - reduced_model: modello ridotto (senza un certo fattore)

    Restituisce (LR_stat, p_value, df).

    Funziona per modelli GLM di statsmodels che espongono:
      - .llf
      - .df_model
    """
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df = full_model.df_model - reduced_model.df_model

    if df <= 0:
        # Nessun grado di libertà aggiuntivo -> LR test non definito
        return lr_stat, np.nan, df

    p_value = chi2.sf(lr_stat, df)
    return lr_stat, p_value, df


# ==========================
# FUNZIONE PRINCIPALE DI ANALISI
# ==========================

def analyze_csv(
    csv_path=DEFAULT_CSV_PATH,
    alpha=ALPHA_DEFAULT,
    H_P=H_P_DEFAULT,
    H_ITER=H_ITER_DEFAULT,
    save_plots=True,
):
    """
    Esegue l'analisi statistica completa sul CSV specificato.

    - Stime di probabilità di successo e iterazioni per (case, version, mode)
    - Calcolo del numero minimo di repliche raccomandato
    - Regressione logistica per la probabilità di successo
    - Modello lineare / ANOVA per le iterazioni (solo successi)
    - Tukey HSD sulle iterazioni tra version
    - Salvataggio grafici su file (se save_plots=True), senza mostrarli a schermo.

    Parameters
    ----------
    csv_path : str
        Percorso del CSV di input.

    alpha : float
        Livello di significatività (per CI e test).

    H_P : float
        Mezzo intervallo target per la proporzione di successo.

    H_ITER : float
        Mezzo intervallo target per la media delle iterazioni (solo successi).

    save_plots : bool
        Se True, salva i grafici in PNG nella cartella corrente.
    """
    print(f"\n=== ANALISI CSV: {csv_path} ===\n")

    # 1) Caricamento dati
    df = pd.read_csv(csv_path)

    # Parsing colonna success in booleano
    df[COL_SUCCESS] = df[COL_SUCCESS].apply(parse_success_column)
    df["success_num"] = df[COL_SUCCESS].astype(int)

    # iterations numerico
    df[COL_ITER] = pd.to_numeric(df[COL_ITER], errors="coerce")

    # 2) DESCRITTIVA PER GRUPPO
    print(">>> Descrittiva per gruppo (case, version, mode)\n")

    grouped = df.groupby(GROUP_COLS)
    results = []

    for group_vals, subdf in grouped:
        group_info = dict(zip(GROUP_COLS, group_vals))

        n_total = len(subdf)
        n_success = subdf[COL_SUCCESS].sum()
        p_hat, p_l, p_u = wilson_ci(n_success, n_total, alpha=alpha)

        n_req_p = required_n_for_proportion(p_hat, H_P, alpha=alpha)
        p_precision_ok = n_total >= n_req_p if not np.isnan(n_req_p) else False

        success_iter = subdf.loc[subdf[COL_SUCCESS], COL_ITER]
        m_iter, s_iter, l_iter, u_iter, n_iter = mean_ci_normal(success_iter, alpha=alpha)

        n_req_iter = required_n_for_mean(s_iter, H_ITER, alpha=alpha)
        iter_precision_ok = (not np.isnan(n_req_iter)) and (n_iter >= n_req_iter)

        row = {
            **group_info,
            "n_total": n_total,
            "n_success": int(n_success),
            "p_hat": p_hat,
            "p_ci_lower": p_l,
            "p_ci_upper": p_u,
            "p_target_halfwidth": H_P,
            "n_required_for_p": n_req_p,
            "sufficient_for_p?": p_precision_ok,
            "n_success_iter": n_iter,
            "mean_iterations_success": m_iter,
            "std_iterations_success": s_iter,
            "iter_ci_lower": l_iter,
            "iter_ci_upper": u_iter,
            "iter_target_halfwidth": H_ITER,
            "n_required_for_iter": n_req_iter,
            "sufficient_for_iter?": iter_precision_ok,
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    sort_cols = [c for c in GROUP_COLS if c in results_df.columns] + ["n_total"]
    results_df = results_df.sort_values(sort_cols)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print(results_df.to_string(index=False))

    out_summary = "risultati_stima_successo_iterazioni.csv"
    results_df.to_csv(out_summary, index=False)
    print(f"\n>>> Risultati descrittivi salvati in: {out_summary}")

    # 3) REGRESSIONE LOGISTICA
    print("\n\n>>> Regressione logistica: probabilità di successo\n")

    formula_logit = f"success_num ~ C({COL_VERSION}) + C({COL_CASE}) + C({COL_MODE})"
    logit_full = smf.glm(
        formula=formula_logit,
        data=df,
        family=sm.families.Binomial()
    ).fit()

    print(logit_full.summary())

    print("\n--- Likelihood Ratio test per ciascun fattore ---")

    # Version
    formula_no_version = f"success_num ~ C({COL_CASE}) + C({COL_MODE})"
    logit_no_version = smf.glm(
        formula=formula_no_version,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_version)
    print(f"Effetto di {COL_VERSION}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # Case
    formula_no_case = f"success_num ~ C({COL_VERSION}) + C({COL_MODE})"
    logit_no_case = smf.glm(
        formula=formula_no_case,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_case)
    print(f"Effetto di {COL_CASE}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # Mode
    formula_no_mode = f"success_num ~ C({COL_VERSION}) + C({COL_CASE})"
    logit_no_mode = smf.glm(
        formula=formula_no_mode,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_mode)
    print(f"Effetto di {COL_MODE}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # 4) MODELLO LINEARE / ANOVA SU ITERAZIONI (SOLO SUCCESSI)
    print("\n\n>>> Modello lineare / ANOVA: iterazioni (solo successi)\n")

    df_success = df[df[COL_SUCCESS]]

    if df_success.shape[0] < 3:
        print("Pochi successi totali: impossibile fare un'ANOVA affidabile.")
    else:
        factors = []

        if df_success[COL_VERSION].nunique() > 1:
            factors.append(f"C({COL_VERSION})")
        if df_success[COL_CASE].nunique() > 1:
            factors.append(f"C({COL_CASE})")
        if df_success[COL_MODE].nunique() > 1:
            factors.append(f"C({COL_MODE})")

        if not factors:
            print(
                "Nei successi tutti i fattori (version/case/mode) hanno un solo livello.\n"
                "Posso stimare solo una media globale delle iterazioni, ma non fare ANOVA."
            )
            print(f"Media iterazioni (successi): {df_success[COL_ITER].mean():.3f}")
        else:
            formula_iter = f"{COL_ITER} ~ " + " + ".join(factors)
            print(f"Formula modello iterazioni: {formula_iter}")
            lm_iter = smf.ols(formula=formula_iter, data=df_success).fit()

            print("\n--- ANOVA (Type II) ---")
            try:
                print(anova_lm(lm_iter, typ=2))
            except ValueError as e:
                print("ANOVA Type II non riuscita (fattori con un solo livello / collinearità?).")
                print(f"Dettaglio errore: {e}")
                print("Mostro comunque il riepilogo del modello lineare.\n")

            print("\n--- Dettaglio modello lineare ---")
            print(lm_iter.summary())

        # Tukey HSD sulle versioni (solo se almeno 2 livelli)
        print("\n\n>>> Confronti multipli (Tukey HSD) sulle iterazioni tra version\n")
        if df_success[COL_VERSION].nunique() < 2:
            print("Tukey non applicabile: nei successi c'è una sola versione.")
        else:
            try:
                tukey_versions = pairwise_tukeyhsd(
                    endog=df_success[COL_ITER],
                    groups=df_success[COL_VERSION],
                    alpha=alpha
                )
                print(tukey_versions.summary())
            except Exception as e:
                print("Errore nel Tukey HSD sulle versioni (forse gruppi troppo piccoli):")
                print(e)

    # 5) GRAFICI (solo salvataggio su file, niente show())
    if save_plots:
        print("\n\n>>> Generazione grafici (salvati su file, non mostrati)\n")

        # Probabilità di successo per version e mode
        prob_by_version_mode = (
            df.groupby([COL_VERSION, COL_MODE])["success_num"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "p_hat", "count": "n"})
        )

        modes = prob_by_version_mode[COL_MODE].unique()
        x_versions = sorted(prob_by_version_mode[COL_VERSION].unique())
        x_pos = np.arange(len(x_versions))
        width = 0.8 / max(1, len(modes))

        plt.figure()
        for i, m in enumerate(modes):
            sub = prob_by_version_mode[prob_by_version_mode[COL_MODE] == m]
            sub = sub.set_index(COL_VERSION).reindex(x_versions)
            plt.bar(x_pos + i * width, sub["p_hat"], width=width, label=str(m))
        plt.xticks(x_pos + width * (len(modes) - 1) / 2, x_versions)
        plt.ylim(0, 1)
        plt.xlabel("Version")
        plt.ylabel("Probabilità di successo (p_hat)")
        plt.title("Probabilità di successo per version e mode")
        plt.legend(title="Mode")
        plt.tight_layout()
        fname = "plot_prob_success_by_version_mode.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Salvato: {fname}")

        # Probabilità di successo per version e case
        prob_by_version_case = (
            df.groupby([COL_VERSION, COL_CASE])["success_num"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "p_hat", "count": "n"})
        )

        cases = sorted(prob_by_version_case[COL_CASE].unique())
        x_pos = np.arange(len(x_versions))
        width = 0.8 / max(1, len(cases))

        plt.figure()
        for i, c in enumerate(cases):
            sub = prob_by_version_case[prob_by_version_case[COL_CASE] == c]
            sub = sub.set_index(COL_VERSION).reindex(x_versions)
            plt.bar(x_pos + i * width, sub["p_hat"], width=width, label=str(c))
        plt.xticks(x_pos + width * (len(cases) - 1) / 2, x_versions)
        plt.ylim(0, 1)
        plt.xlabel("Version")
        plt.ylabel("Probabilità di successo (p_hat)")
        plt.title("Probabilità di successo per version e case")
        plt.legend(title="Case")
        plt.tight_layout()
        fname = "plot_prob_success_by_version_case.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Salvato: {fname}")

        # Boxplot iterazioni (solo successi) per version
        if df_success.shape[0] >= 3:
            plt.figure()
            data_to_plot = [
                df_success.loc[df_success[COL_VERSION] == v, COL_ITER].dropna()
                for v in x_versions
            ]
            plt.boxplot(data_to_plot, labels=x_versions)
            plt.xlabel("Version")
            plt.ylabel("Iterazioni (solo successi)")
            plt.title("Distribuzione iterazioni (successi) per version")
            plt.tight_layout()
            fname = "plot_iterations_boxplot_by_version.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"Salvato: {fname}")

            # Boxplot iterazioni per version separato per mode (se > 1 livello)
            modes_success = sorted(df_success[COL_MODE].unique())
            for m in modes_success:
                sub = df_success[df_success[COL_MODE] == m]
                if sub.shape[0] < 3:
                    continue
                plt.figure()
                data_to_plot = [
                    sub.loc[sub[COL_VERSION] == v, COL_ITER].dropna()
                    for v in x_versions
                ]
                plt.boxplot(data_to_plot, labels=x_versions)
                plt.xlabel("Version")
                plt.ylabel("Iterazioni (solo successi)")
                plt.title(f"Iterazioni (successi) per version - mode={m}")
                plt.tight_layout()
                fname = f"plot_iterations_boxplot_by_version_mode_{m}.png"
                plt.savefig(fname, dpi=150)
                plt.close()
                print(f"Salvato: {fname}")
        else:
            print("Non ci sono abbastanza successi per grafici sulle iterazioni.")

    print("\n=== Analisi completata ===\n")


# ==========================
# MAIN (per uso da CLI)
# ==========================

def main():
    """
    Esegue l'analisi sul CSV di default 'dati_processo.csv'.

    In produzione, tipicamente:
      1) chiami CSVgen(rows, 'dati_processo.csv') nel tuo codice di raccolta dati
      2) poi lanci questo script (o importi analyze_csv) per l'analisi.
    """
    analyze_csv(DEFAULT_CSV_PATH)


if __name__ == "__main__":
    main()