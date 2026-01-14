import os
import json
import csv
import argparse
from typing import Set, Tuple, List, Dict, Any


# Nome di default del file di configurazione
CONFIG_FILE = "configs/config_irius.json"

# Colonne dei CSV IriusRisk
THREAT_COLUMN = "Threat"
COMPONENT_COLUMN = "Component"


# ==========================
#  FUNZIONI DI UTILITÀ BASE
# ==========================

def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Legge il file di configurazione JSON e restituisce un dizionario con:
      - case_studies: lista di nomi di case study
      - output_dir: cartella di output
      - irius_root: cartella root dei case study (default: 'IriusRisk')

    Supporta sia:
      { "case_studies": ["cs1", "cs2"], ... }
    sia:
      { "case_study": "cs1", ... }
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"File di configurazione '{config_path}' non trovato. "
            f"Crealo con almeno 'case_studies' (o 'case_study') e 'output_dir'."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "output_dir" not in cfg:
        raise KeyError("Il file di configurazione deve contenere la chiave 'output_dir'.")

    # Determina la lista di case_studies
    case_studies: List[str] = []
    if "case_studies" in cfg:
        cs = cfg["case_studies"]
        if isinstance(cs, str):
            case_studies = [cs]
        elif isinstance(cs, list):
            case_studies = [str(x) for x in cs if str(x).strip()]
        else:
            raise TypeError("'case_studies' deve essere una stringa o una lista di stringhe.")
    elif "case_study" in cfg:
        case_studies = [str(cfg["case_study"])]
    else:
        raise KeyError("Il file di configurazione deve contenere 'case_studies' o 'case_study'.")

    if not case_studies:
        raise ValueError("La lista dei case_study è vuota.")

    output_dir = cfg["output_dir"]
    irius_root = cfg.get("irius_root", "IriusRisk")

    return {
        "case_studies": case_studies,
        "output_dir": output_dir,
        "irius_root": irius_root,
    }


def load_threats_from_csv(
    path: str,
    threat_column: str = THREAT_COLUMN,
    component_column: str = COMPONENT_COLUMN,
) -> Set[Tuple[str, str]]:
    """
    Restituisce l'insieme dei threat come coppie (component, threat),
    normalizzate (strip + lowercase).
    """
    threats: Set[Tuple[str, str]] = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing_cols = [
            col for col in (component_column, threat_column) if col not in fieldnames
        ]
        if missing_cols:
            raise ValueError(
                f"Nel file '{path}' mancano le colonne {missing_cols}. "
                f"Colonne trovate: {fieldnames}"
            )

        for row in reader:
            comp = row.get(component_column, "")
            thr = row.get(threat_column, "")
            if comp is None or thr is None:
                continue

            comp_norm = comp.strip().lower()
            thr_norm = thr.strip().lower()

            if comp_norm and thr_norm:
                 threats.add((comp_norm, thr_norm))
            #threats.add((thr_norm))

    return threats


def find_reference_tm(manual_dir: str) -> str:
    """
    Cerca nel folder Manual il file CSV il cui nome termina con 'human.csv'.
    """
    if not os.path.isdir(manual_dir):
        raise FileNotFoundError(f"La cartella '{manual_dir}' non esiste.")

    candidate_files: List[str] = []
    for fname in os.listdir(manual_dir):
        if fname.lower().endswith("human.csv"):
            candidate_files.append(os.path.join(manual_dir, fname))

    if not candidate_files:
        raise FileNotFoundError(
            f"Nella cartella '{manual_dir}' non è stato trovato un file '*human.csv'."
        )
    if len(candidate_files) > 1:
        raise RuntimeError(
            "Trovati più file '*human.csv' in '{manual_dir}':\n" +
            "\n".join(candidate_files)
        )

    return candidate_files[0]


def find_generated_tms(llm_dir: str) -> List[str]:
    """
    Restituisce la lista dei CSV in LLM-Based-TM (escludendo eventuali '*human.csv').
    """
    if not os.path.isdir(llm_dir):
        raise FileNotFoundError(f"La cartella '{llm_dir}' non esiste.")

    files: List[str] = []
    for fname in os.listdir(llm_dir):
        if fname.lower().endswith(".csv") and not fname.lower().endswith("human.csv"):
            files.append(os.path.join(llm_dir, fname))

    if not files:
        raise FileNotFoundError(
            f"Nella cartella '{llm_dir}' non è stato trovato nessun generated TM."
        )

    return sorted(files)


def safe_ratio(num: int, den: int) -> float:
    return (num / den) if den != 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """
    F_beta generico; con beta=2 ottieni F2.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


# ==========================
#  LOGICA DI CONFRONTO
# ==========================

def evaluate_case_study(
    irius_root: str,
    case_study: str,
    starting_index: int = 1,
) -> Dict[str, Any]:
    """
    Esegue tutte le operazioni di confronto per un singolo case_study.

    Ritorna un dizionario con:
      - per_file_metrics: lista di metriche numeriche per ogni generated TM
      - results_rows: lista di righe pronte per il CSV
      - last_index: ultimo indice usato (per continuare la numerazione globale)
    """
    print(f"\n[CASE_STUDY] {case_study}")

    manual_dir = os.path.join(irius_root, case_study, "Manual")
    llm_dir = os.path.join(irius_root, case_study, "LLM-Based-TM")

    print(f"[PATH] Manual dir : {manual_dir}")
    print(f"[PATH] LLM dir    : {llm_dir}")

    # Reference TM
    reference_path = find_reference_tm(manual_dir)
    print(f"[INFO] Reference TM: {reference_path}")
    ref_threats = load_threats_from_csv(reference_path)

    # Generated TM
    generated_files = find_generated_tms(llm_dir)
    print(f"[INFO] Trovati {len(generated_files)} generated TM per '{case_study}'.")

    per_file_metrics: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    current_index = starting_index

    for gen_path in generated_files:
        gen_threats = load_threats_from_csv(gen_path)

        tp = len(gen_threats & ref_threats)
        fp = len(gen_threats - ref_threats)
        fn = len(ref_threats - gen_threats)
        support = tp + fn  # positivi reali (nel reference TM)

        recall = safe_ratio(tp, tp + fn)
        precision = safe_ratio(tp, tp + fp)
        f1 = f1_score(precision, recall)
        f2 = f_beta_score(precision, recall, beta=2.0)

        # Accuracy sul dominio ref ∪ gen: TN non osservabili -> assumiamo TN=0
        denom_acc = tp + fp + fn
        accuracy = safe_ratio(tp, denom_acc)

        # TN sconosciuti -> ipotizziamo TN = 0
        tn = 0
        specificity = safe_ratio(tn, tn + fp)          # ≈ 0 se ci sono FP
        balanced_accuracy = (recall + specificity) / 2 # ≈ recall/2

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "f2": f2,
            "accuracy": accuracy,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        }
        per_file_metrics.append(metrics)

        row = {
            "Index": current_index,
            "case_study": case_study,
            "FILE": os.path.basename(gen_path),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "support": support,
            "recall": f"{recall:.4f}",
            "precision": f"{precision:.4f}",
            "F1-score": f"{f1:.4f}",
            "F2-score": f"{f2:.4f}",
            "specificity": f"{specificity:.4f}",
            "balanced_accuracy": f"{balanced_accuracy:.4f}",
            "accuracy": f"{accuracy:.4f}",
        }
        results_rows.append(row)

        print(
            f"[INFO] {current_index}. {case_study} / {os.path.basename(gen_path)} -> "
            f"TP={tp}, FP={fp}, FN={fn}, support={support}, "
            f"recall={recall:.4f}, precision={precision:.4f}, "
            f"F1={f1:.4f}, F2={f2:.4f}, "
            f"specificity={specificity:.4f}, "
            f"balanced_acc={balanced_accuracy:.4f}, "
            f"accuracy={accuracy:.4f}"
        )

        current_index += 1

    return {
        "per_file_metrics": per_file_metrics,
        "results_rows": results_rows,
        "last_index": current_index - 1,
    }


def process_all_case_studies(
    irius_root: str,
    case_studies: List[str],
    starting_index: int = 1,
) -> Dict[str, Any]:
    """
    Orchetra l'elaborazione di tutti i case_studies.

    Ritorna:
      - all_per_file_metrics: lista aggregata di metriche per tutti i file
      - all_results_rows: tutte le righe da scrivere nel CSV
    """
    all_per_file_metrics: List[Dict[str, Any]] = []
    all_results_rows: List[Dict[str, Any]] = []

    global_index = starting_index

    for case_study in case_studies:
        result = evaluate_case_study(
            irius_root=irius_root,
            case_study=case_study,
            starting_index=global_index,
        )

        per_file_metrics = result["per_file_metrics"]
        results_rows = result["results_rows"]
        last_index = result["last_index"]

        all_per_file_metrics.extend(per_file_metrics)
        all_results_rows.extend(results_rows)

        global_index = last_index + 1

    return {
        "all_per_file_metrics": all_per_file_metrics,
        "all_results_rows": all_results_rows,
    }


# ==========================
#  METRICHE GLOBALI & OUTPUT
# ==========================

def compute_global_metrics(per_file_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcola metriche globali (macro, micro, weighted) a partire dalla lista
    di metriche per-file.
    """
    n = len(per_file_metrics)
    if n == 0:
        return {}

    macro_precision = sum(m["precision"] for m in per_file_metrics) / n
    macro_recall = sum(m["recall"] for m in per_file_metrics) / n
    macro_f1 = sum(m["f1"] for m in per_file_metrics) / n
    macro_f2 = sum(m["f2"] for m in per_file_metrics) / n
    macro_specificity = sum(m["specificity"] for m in per_file_metrics) / n
    macro_bal_acc = sum(m["balanced_accuracy"] for m in per_file_metrics) / n

    total_support = sum(m["support"] for m in per_file_metrics)
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_file_metrics) / total_support
        if total_support > 0 else 0.0
    )

    total_tp = sum(m["tp"] for m in per_file_metrics)
    total_fp = sum(m["fp"] for m in per_file_metrics)
    total_fn = sum(m["fn"] for m in per_file_metrics)

    micro_precision = safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = safe_ratio(total_tp, total_tp + total_fn)
    micro_f1 = f1_score(micro_precision, micro_recall)
    micro_f2 = f_beta_score(micro_precision, micro_recall, beta=2.0)

    # TN aggregato = 0 per coerenza con l'ipotesi
    micro_specificity = 0.0
    micro_bal_acc = (micro_recall + micro_specificity) / 2.0

    micro_accuracy = safe_ratio(total_tp, total_tp + total_fp + total_fn)

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_f2": macro_f2,
        "macro_specificity": macro_specificity,
        "macro_bal_acc": macro_bal_acc,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_f2": micro_f2,
        "micro_specificity": micro_specificity,
        "micro_bal_acc": micro_bal_acc,
        "micro_accuracy": micro_accuracy,
    }


def print_global_metrics(global_metrics: Dict[str, float]) -> None:
    """
    Stampa in console le metriche globali.
    """
    if not global_metrics:
        print("Nessun file generated TM trovato, niente metriche globali.")
        return

    print("\n[GLOBAL METRICS - ACROSS ALL CASE_STUDIES]")
    print(f"Macro Precision      : {global_metrics['macro_precision']:.4f}")
    print(f"Macro Recall         : {global_metrics['macro_recall']:.4f}")
    print(f"Macro F1             : {global_metrics['macro_f1']:.4f}")
    print(f"Macro F2             : {global_metrics['macro_f2']:.4f}")
    print(f"Macro Specificity(*) : {global_metrics['macro_specificity']:.4f}")
    print(f"Macro Balanced Acc(*): {global_metrics['macro_bal_acc']:.4f}")
    print(f"Weighted F1          : {global_metrics['weighted_f1']:.4f}")
    print(f"Micro Precision      : {global_metrics['micro_precision']:.4f}")
    print(f"Micro Recall         : {global_metrics['micro_recall']:.4f}")
    print(f"Micro F1             : {global_metrics['micro_f1']:.4f}")
    print(f"Micro F2             : {global_metrics['micro_f2']:.4f}")
    print(f"Micro Spec. (*)      : {global_metrics['micro_specificity']:.4f}")
    print(f"Micro Bal. Acc. (*)  : {global_metrics['micro_bal_acc']:.4f}")
    print(f"Accuracy             : {global_metrics['micro_accuracy']:.4f}")
    print("\n(*) Specificity e balanced accuracy calcolate assumendo TN = 0 "
          "per mancanza di veri negativi osservabili.")


def write_results_csv(output_file: str, rows: List[Dict[str, Any]]) -> None:
    """
    Scrive il CSV dei risultati per-file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Index", "case_study", "FILE",
                "TP", "FP", "FN",
                "support", "recall", "precision",
                "F1-score", "F2-score",
                "specificity", "balanced_accuracy",
                "accuracy"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] File CSV scritto in: {output_file}")


# ==========================
#  MAIN + ENTRY POINT
# ==========================

def main() -> None:
    """
    Funzione main:
      - gestisce gli argomenti da linea di comando,
      - carica la configurazione,
      - processa i case_studies,
      - calcola e stampa le metriche globali,
      - scrive il CSV.
    """
    parser = argparse.ArgumentParser(
        description="Confronto batch di threat models IriusRisk rispetto a reference human TM."
    )
    parser.add_argument(
        "-c", "--config",
        default=CONFIG_FILE,
        help=f"Path del file di configurazione (default: {CONFIG_FILE})"
    )
    args = parser.parse_args()

    config_path = args.config
    cfg = load_config(config_path)

    case_studies: List[str] = cfg["case_studies"]
    output_dir: str = cfg["output_dir"]
    irius_root: str = cfg["irius_root"]

    output_file = os.path.join(output_dir, "dati_Irius.csv")

    print(f"[CONFIG] Config file : {config_path}")
    print(f"[CONFIG] Irius root  : {irius_root}")
    print(f"[CONFIG] Case studies: {case_studies}")
    print(f"[CONFIG] Output file : {output_file}")

    processed = process_all_case_studies(
        irius_root=irius_root,
        case_studies=case_studies,
        starting_index=1,
    )

    all_per_file_metrics = processed["all_per_file_metrics"]
    all_results_rows = processed["all_results_rows"]

    global_metrics = compute_global_metrics(all_per_file_metrics)
    print_global_metrics(global_metrics)
    write_results_csv(output_file, all_results_rows)


if __name__ == "__main__":
    main()