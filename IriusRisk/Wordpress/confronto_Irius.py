import os
import csv
import pprint
from typing import Set, Tuple, List, Dict, Any


MANUAL_DIR = "Manual"
LLM_DIR = "LLM-Based-TM"
OUTPUT_FILE = "dati_Irius.csv"
THREAT_COLUMN = "Threat"
COMPONENT_COLUMN = "Component"


def load_threats_from_csv(
    path: str,
    threat_column: str = THREAT_COLUMN,
    component_column: str = COMPONENT_COLUMN,
) -> Set[Tuple[str, str]]:
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

    return threats


def find_reference_tm(manual_dir: str = MANUAL_DIR) -> str:
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
            "Trovati più file '*human.csv':\n" + "\n".join(candidate_files)
        )

    return candidate_files[0]


def find_generated_tms(llm_dir: str = LLM_DIR) -> List[str]:
    if not os.path.isdir(llm_dir):
        raise FileNotFoundError(f"La cartella '{llm_dir}' non esiste.")

    files = []
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


def main():
    reference_path = find_reference_tm(MANUAL_DIR)
    print(f"[INFO] Reference TM: {reference_path}")
    ref_threats = load_threats_from_csv(reference_path)

    generated_files = find_generated_tms(LLM_DIR)
    print(f"[INFO] Trovati {len(generated_files)} generated TM.")

    per_file_metrics: List[Dict[str, Any]] = []
    results_rows = []

    for idx, gen_path in enumerate(generated_files, start=1):
        gen_threats = load_threats_from_csv(gen_path)

        tp = len(gen_threats & ref_threats)
        fp = len(gen_threats - ref_threats)
        fn = len(ref_threats - gen_threats)
        support = tp + fn

        recall = safe_ratio(tp, tp + fn)
        precision = safe_ratio(tp, tp + fp)
        f1 = f1_score(precision, recall)
        accuracy = safe_ratio(tp, tp + fp + fn)

        per_file_metrics.append({
            "tp": tp, "fp": fp, "fn": fn,
            "support": support,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy,
        })

        results_rows.append({
            "Index": idx,
            "FILE": os.path.basename(gen_path),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "support": support,
            "recall": f"{recall:.4f}",
            "precision": f"{precision:.4f}",
            "F1-score": f"{f1:.4f}",
            "accuracy": f"{accuracy:.4f}",
        })

        print(
            f"[INFO] {idx}. {os.path.basename(gen_path)} -> "
            f"TP={tp}, FP={fp}, FN={fn}, support={support}, "
            f"recall={recall:.4f}, precision={precision:.4f}, "
            f"F1={f1:.4f}, accuracy={accuracy:.4f}"
        )

    # ---------------------------
    # GLOBAL METRICS (SOLO OUTPUT)
    # ---------------------------
    print("\n[GLOBAL METRICS]")

    n = len(per_file_metrics)

    macro_precision = sum(m["precision"] for m in per_file_metrics) / n
    macro_recall = sum(m["recall"] for m in per_file_metrics) / n
    macro_f1 = sum(m["f1"] for m in per_file_metrics) / n

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
    micro_accuracy = safe_ratio(total_tp, total_tp + total_fp + total_fn)

    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall   : {micro_recall:.4f}")
    print(f"Micro F1       : {micro_f1:.4f}")
    print(f"Accuracy       : {micro_accuracy:.4f}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Index", "FILE", "TP", "FP", "FN",
                        "support", "recall", "precision", "F1-score", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\n[OK] File CSV scritto in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()