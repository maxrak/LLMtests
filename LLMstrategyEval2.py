import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm, chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt


# ==========================
# CONFIGURAZIONE
# ==========================

DEFAULT_CSV_PATH = "dati_processo.csv"

COL_CASE = "case"
COL_VERSION = "version"
COL_MODE = "mode"
COL_RAG = "RAG"
COL_SUCCESS = "success"
COL_ITER = "iterations"

# gruppi per la tabella descrittiva
GROUP_COLS = [COL_CASE, COL_VERSION, COL_MODE, COL_RAG]

# parametri statistici
ALPHA_DEFAULT = 0.20           # livello di confidenza / significatività
H_P_DEFAULT = 0.20            # ampiezza target ± per p (probabilità di successo)
H_ITER_DEFAULT = 4.0           # ampiezza target ± per media iterazioni (successi)


# ==========================
# FUNZIONI DI SUPPORTO
# ==========================

def parse_success_column(s):
    """
    Converte la colonna di successo in booleano (True = successo, False = fallimento).
    Gestisce 0/1, True/False, stringhe.
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
    Restituisce (p_hat, lower, upper).
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)

    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / (2 * n)
    centre = p_hat + z**2 / (2 * n)
    margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return p_hat, lower, upper


def required_n_for_proportion(p_hat, h, alpha=0.05):
    """
    Numero minimo di campioni per ottenere un IC 95% di ampiezza ±h su p.
    """
    if np.isnan(p_hat):
        p = 0.5
    else:
        p = p_hat
    z = norm.ppf(1 - alpha / 2)
    n_req = (z**2 * p * (1 - p)) / (h**2)
    return n_req


def mean_ci_normal(x, alpha=0.05):
    """
    Intervallo di confidenza per la media (approssimazione normale / CLT).
    Restituisce (mean, std, lower, upper, n).
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
    Numero minimo di campioni per stimare la media con IC ±h, data la deviazione standard s.
    """
    if s is None or np.isnan(s) or s == 0:
        return np.nan
    z = norm.ppf(1 - alpha / 2)
    n_req = (z * s / h)**2
    return n_req


def lr_test(full_model, reduced_model):
    """
    Likelihood Ratio test tra modello completo e ridotto.
    Restituisce (LR_stat, p_value, df).
    """
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df = full_model.df_model - reduced_model.df_model

    if df <= 0:
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
    save_plots=False,
):
    """
    Analisi DOE con fattori: version, case, mode, RAG.

    - Stima p(success) per ogni combinazione (case,version,mode,RAG)
    - IC di Wilson e check sufficienza n (H_P)
    - Media iterazioni sui successi, IC e check sufficienza n (H_ITER)
    - Regressione logistica: success ~ version + case + mode + RAG
    - LR test per ogni fattore
    - Modello lineare / ANOVA sulle iterazioni (solo successi)
    - Tukey HSD sulle version per le iterazioni
    - (opzionale) grafici marginali salvati su file.
    """

    print(f"\n=== ANALISI DOE SU FILE: {csv_path} ===\n")

    # 1) Caricamento
    df = pd.read_csv(csv_path)

    # parsing success
    df[COL_SUCCESS] = df[COL_SUCCESS].apply(parse_success_column)
    df["success_num"] = df[COL_SUCCESS].astype(int)

    # iterations numerico
    df[COL_ITER] = pd.to_numeric(df[COL_ITER], errors="coerce")

    # 2) DESCRITTIVA PER GRUPPO
    print(">>> Descrittiva per gruppo (case, version, mode, RAG)\n")

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
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(results_df.to_string(index=False))

    out_summary = "risultati_DOE_descrittiva.csv"
    results_df.to_csv(out_summary, index=False)
    print(f"\n>>> Risultati descrittivi salvati in: {out_summary}")

    # 3) REGRESSIONE LOGISTICA
    print("\n\n>>> Regressione logistica: probabilità di successo\n")

    # assicuriamoci che mode e RAG siano stringhe/categorical friendly
    df[COL_MODE] = df[COL_MODE].astype(str)
    df[COL_RAG] = df[COL_RAG].astype(str)
    df[COL_VERSION] = df[COL_VERSION].astype(str)
    df[COL_CASE] = df[COL_CASE].astype(str)

    logit_formula = (
        f"success_num ~ C({COL_VERSION}) + C({COL_CASE}) + C({COL_MODE}) + C({COL_RAG})"
    )
    logit_full = smf.glm(
        formula=logit_formula,
        data=df,
        family=sm.families.Binomial()
    ).fit()

    print(logit_full.summary())

    print("\n--- Likelihood Ratio test per ciascun fattore (version, case, mode, RAG) ---")

    # senza VERSION
    formula_no_version = f"success_num ~ C({COL_CASE}) + C({COL_MODE}) + C({COL_RAG})"
    logit_no_version = smf.glm(
        formula=formula_no_version,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_version)
    print(f"Effetto di {COL_VERSION}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # senza CASE
    formula_no_case = f"success_num ~ C({COL_VERSION}) + C({COL_MODE}) + C({COL_RAG})"
    logit_no_case = smf.glm(
        formula=formula_no_case,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_case)
    print(f"Effetto di {COL_CASE}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # senza MODE
    formula_no_mode = f"success_num ~ C({COL_VERSION}) + C({COL_CASE}) + C({COL_RAG})"
    logit_no_mode = smf.glm(
        formula=formula_no_mode,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_mode)
    print(f"Effetto di {COL_MODE}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # senza RAG
    formula_no_rag = f"success_num ~ C({COL_VERSION}) + C({COL_CASE}) + C({COL_MODE})"
    logit_no_rag = smf.glm(
        formula=formula_no_rag,
        data=df,
        family=sm.families.Binomial()
    ).fit()
    lr_stat, lr_pvalue, lr_df = lr_test(logit_full, logit_no_rag)
    print(f"Effetto di {COL_RAG}: LR={lr_stat:.3f}, df={int(lr_df)}, p={lr_pvalue:.4g}")

    # 4) MODELLO LINEARE / ANOVA SULLE ITERAZIONI (SOLO SUCCESSI)
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
        if df_success[COL_RAG].nunique() > 1:
            factors.append(f"C({COL_RAG})")

        if not factors:
            print(
                "Nei successi tutti i fattori (version/case/mode/RAG) hanno un solo livello.\n"
                "Posso stimare solo una media globale delle iterazioni."
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

        # Tukey HSD sulle version per iterazioni
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

    # 5) GRAFICI (OPZIONALE)
    if save_plots:
        print("\n\n>>> Generazione grafici marginali (salvati su file)\n")

        versions = sorted(df[COL_VERSION].unique())
        modes = sorted(df[COL_MODE].unique())
        cases = sorted(df[COL_CASE].unique())
        rags = sorted(df[COL_RAG].unique())

        # --- Marginale per VERSION ---
        grid_v = pd.DataFrame({
            COL_VERSION: np.repeat(versions, len(cases)*len(modes)*len(rags)),
            COL_CASE:   np.tile(np.repeat(cases, len(modes)*len(rags)), len(versions)),
            COL_MODE:   np.tile(np.repeat(modes, len(rags)), len(versions)*len(cases)),
            COL_RAG:    np.tile(rags, len(versions)*len(cases)*len(modes)),
        })
        grid_v["success_num"] = 0
        grid_v["p_pred"] = logit_full.predict(grid_v)
        marginal_version = grid_v.groupby(COL_VERSION)["p_pred"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(marginal_version[COL_VERSION], marginal_version["p_pred"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Version")
        ax.set_ylabel("Predicted P(success)")
        ax.set_title("Marginal predicted success probability by version")
        fname = "imgs/marginal_p_by_version.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")

        # --- Marginale per MODE ---
        grid_m = pd.DataFrame({
            COL_MODE:   np.repeat(modes, len(cases)*len(versions)*len(rags)),
            COL_CASE:   np.tile(np.repeat(cases, len(versions)*len(rags)), len(modes)),
            COL_VERSION: np.tile(np.repeat(versions, len(rags)), len(modes)*len(cases)),
            COL_RAG:    np.tile(rags, len(modes)*len(cases)*len(versions)),
        })
        grid_m["success_num"] = 0
        grid_m["p_pred"] = logit_full.predict(grid_m)
        marginal_mode = grid_m.groupby(COL_MODE)["p_pred"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(marginal_mode[COL_MODE], marginal_mode["p_pred"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Predicted P(success)")
        ax.set_title("Marginal predicted success probability by mode")
        fname = "imgs/marginal_p_by_mode.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")

        # --- Marginale per CASE ---
        grid_c = pd.DataFrame({
            COL_CASE:   np.repeat(cases, len(versions)*len(modes)*len(rags)),
            COL_VERSION: np.tile(np.repeat(versions, len(modes)*len(rags)), len(cases)),
            COL_MODE:   np.tile(np.repeat(modes, len(rags)), len(cases)*len(versions)),
            COL_RAG:    np.tile(rags, len(cases)*len(versions)*len(modes)),
        })
        grid_c["success_num"] = 0
        grid_c["p_pred"] = logit_full.predict(grid_c)
        marginal_case = grid_c.groupby(COL_CASE)["p_pred"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(marginal_case[COL_CASE], marginal_case["p_pred"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Case")
        ax.set_ylabel("Predicted P(success)")
        ax.set_title("Marginal predicted success probability by case")
        plt.xticks(rotation=45, ha="right")
        fname = "imgs/marginal_p_by_case.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")

        # --- Marginale per RAG (QUI LA PARTE CHE TI MANCA) ---
        grid_r = pd.DataFrame({
            COL_RAG:     np.repeat(rags, len(cases)*len(versions)*len(modes)),
            COL_CASE:    np.tile(np.repeat(cases, len(versions)*len(modes)), len(rags)),
            COL_VERSION: np.tile(np.repeat(versions, len(modes)), len(rags)*len(cases)),
            COL_MODE:    np.tile(modes, len(rags)*len(cases)*len(versions)),
        })
        grid_r["success_num"] = 0
        grid_r["p_pred"] = logit_full.predict(grid_r)
        marginal_rag = grid_r.groupby(COL_RAG)["p_pred"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(marginal_rag[COL_RAG], marginal_rag["p_pred"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("RAG")
        ax.set_ylabel("Predicted P(success)")
        ax.set_title("Marginal predicted success probability by RAG")
        fname = "imgs/marginal_p_by_RAG.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")

        # --- OPTIONAL: Version × RAG (line plot) ---
        print("Creo anche il plot Version × RAG...")
        # usiamo grid_v di prima ma raggruppiamo per version e RAG
        v_r = grid_v.groupby([COL_VERSION, COL_RAG])["p_pred"].mean().reset_index()

        fig, ax = plt.subplots()
        for rag_level in rags:
            sub = v_r[v_r[COL_RAG] == rag_level]
            ax.plot(sub[COL_VERSION], sub["p_pred"], marker="o", label=f"RAG={rag_level}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Version")
        ax.set_ylabel("Predicted P(success)")
        ax.set_title("Predicted success probability by Version and RAG")
        ax.legend()
        fname = "imgs/p_by_version_and_RAG.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")
                # =====================================================
        # GRAFICI ADDIZIONALI RICHIESTI
        #   1) Probabilità di successo per VERSION e CASE
        #   2) Probabilità di successo per RAG e CASE
        #   3) Probabilità di successo per MODE e VERSION
        # =====================================================

        # 1) Probabilità di successo per VERSION e CASE
        print("Creo grafici: P(success) per version e case...")
        prob_v_case = (
            df.groupby([COL_CASE, COL_VERSION])["success_num"]
              .mean()
              .reset_index()
              .rename(columns={"success_num": "p_success"})
        )

        for c in cases:
            sub = prob_v_case[prob_v_case[COL_CASE] == c]
            if sub.empty:
                continue
            fig, ax = plt.subplots()
            ax.bar(sub[COL_VERSION], sub["p_success"])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Version")
            ax.set_ylabel("P(success)")
            ax.set_title(f"P(success) per version - case={c}")
            fname = f"imgs/prob_success_by_version_case_{c}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Salvato: {fname}")

        # 2) Probabilità di successo per RAG e CASE
        print("Creo grafici: P(success) per RAG e case...")
        prob_r_case = (
            df.groupby([COL_CASE, COL_RAG])["success_num"]
              .mean()
              .reset_index()
              .rename(columns={"success_num": "p_success"})
        )

        for c in cases:
            sub = prob_r_case[prob_r_case[COL_CASE] == c]
            if sub.empty:
                continue
            fig, ax = plt.subplots()
            ax.bar(sub[COL_RAG], sub["p_success"])
            ax.set_ylim(0, 1)
            ax.set_xlabel("RAG")
            ax.set_ylabel("P(success)")
            ax.set_title(f"P(success) per RAG - case={c}")
            fname = f"imgs/prob_success_by_RAG_case_{c}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Salvato: {fname}")

        # 3) Probabilità di successo per MODE e VERSION
        print("Creo grafico: P(success) per mode e version...")
        prob_v_mode = (
            df.groupby([COL_VERSION, COL_MODE])["success_num"]
              .mean()
              .reset_index()
              .rename(columns={"success_num": "p_success"})
        )

        fig, ax = plt.subplots()
        # grafico a barre raggruppate: x = version, barre = mode
        modes_unique = sorted(prob_v_mode[COL_MODE].unique())
        x = np.arange(len(versions))
        width = 0.8 / max(1, len(modes_unique))

        for i, m in enumerate(modes_unique):
            sub = prob_v_mode[prob_v_mode[COL_MODE] == m]
            sub = sub.set_index(COL_VERSION).reindex(versions)
            ax.bar(x + i * width, sub["p_success"], width=width, label=str(m))

        ax.set_xticks(x + width * (len(modes_unique) - 1) / 2)
        ax.set_xticklabels(versions)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Version")
        ax.set_ylabel("P(success)")
        ax.set_title("P(success) per mode e version")
        ax.legend(title="Mode")
        fname = "imgs/prob_success_by_mode_version.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Salvato: {fname}")

    print("\n=== Analisi DOE completata ===\n")


# ==========================
# MAIN
# ==========================

def main():
    analyze_csv(DEFAULT_CSV_PATH, save_plots=False)


if __name__ == "__main__":
    main()