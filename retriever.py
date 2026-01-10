import requests
import json
import csv
from pathlib import Path
from pprint import pprint

# ========= PARAMETRI =========

# Percorso del file JSON di input (modifica se serve)
INPUT_JSON = Path("res.json")          # o "res.txt" se il tuo file ha quel nome
# Percorso del CSV di output (questo è quello che userai poi nell'analisi)
OUTPUT_CSV = Path("dati_processo.csv")

# Nome delle colonne del CSV (compatibili con lo script di analisi)
CSV_FIELDNAMES = ["case", "version", "mode", "success", "iterations","rag"]

def sessions():
    url = "https://database.vseclab.lan/api/v2/tables/mmzjs8j9yb88tsf/records?offset=0&limit=25&where=&viewId=vwpkfrf4ivtd846e"

    headers = {"xc-token": "7x4DdSHBSIeVKVv_sRHW7Dz_VJ1bf8SkXzRJMLz0"}

    page=1
    rows=[]
    alldone=False

    while (not alldone):
        params = {
            "page": page,
            "pageSize": 25   # oppure 500, 1000 se consentito
        }
        response = requests.get(url, headers=headers, params=params, verify="public.crt")
        data=response.json()
        rows.extend(data["list"])
        #print(json.dumps(data, indent=2, ensure_ascii=False))
        page+=1
        if data["pageInfo"]["isLastPage"]:
            alldone=True

    return rows

import csv

# campi richiesti per l'analisi
COL_CASE = "case"
COL_VERSION = "version"
COL_MODE = "mode"
COL_SUCCESS = "success"
COL_ITER = "iterations"
COL_RAG = "RAG"
COL_CREATED = "created at"
COL_UPDATED = "updated at"

CSV_FIELDNAMES = [COL_CASE, COL_VERSION, COL_MODE, COL_RAG, COL_SUCCESS, COL_ITER,COL_CREATED,COL_UPDATED]


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
            macm = rec.get("generated_macms",None)
            rag =  rec.get("rag",None)
            created = rec.get("created_at",None)
            updated = rec.get("updated_at", None)
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

            if success == macm :
                writer.writerow({
                    COL_CASE: case,
                    COL_VERSION: version,
                    COL_MODE: mode,
                    COL_RAG : rag,
                    COL_SUCCESS: success,
                    COL_ITER: iterations,
                    COL_CREATED: created,
                    COL_UPDATED: updated

                })

    return output_csv_path


def test():
    rows=sessions()
    #pprint(rows)
    CSVgen(rows, "dati_processo.csv")

test()