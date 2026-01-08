#!/usr/bin/env python3
# ci_iterations.py

import argparse
import math
import sqlite3
from typing import Optional

import pandas as pd
from scipy import stats


def mean_ci_t(values, conf: float = 0.95):
    """IC della media con t di Student."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = int(values.shape[0])
    if n < 2:
        return {
            "n": n,
            "mean": float(values.mean()) if n == 1 else float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    alpha = 1.0 - conf
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * se
    return {
        "n": n,
        "mean": mean,
        "std": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def run(db_path: str, table: str, value_col: str, conf: float,
        where: Optional[str] = None, limit: Optional[str] = None, groupby: Optional[str] = None):
    conn = sqlite3.connect(db_path)

    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"

    print(sql)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    if value_col not in df.columns:
        raise SystemExit(f"Colonna '{value_col}' non trovata. Colonne disponibili: {list(df.columns)}")

    if groupby:
        if groupby not in df.columns:
            raise SystemExit(f"Colonna di gruppo '{groupby}' non trovata. Colonne disponibili: {list(df.columns)}")

        rows = []
        for key, g in df.groupby(groupby, dropna=False):
            r = mean_ci_t(g[value_col], conf=conf)
            r[groupby] = key
            rows.append(r)

        out = pd.DataFrame(rows)[[groupby, "n", "mean", "std", "ci_low", "ci_high"]].sort_values(
            by=groupby, kind="stable"
        )
        print(out.to_string(index=False))
    else:
        r = mean_ci_t(df[value_col], conf=conf)
        print(f"Tabella: {table} | Colonna: {value_col}")
        if where:
            print(f"Filtro WHERE: {where}")
        print(f"n={r['n']}  mean={r['mean']:.6g}  std={r['std']:.6g}")
        dim=r['ci_high']-r['ci_low']
        print(f"IC {conf*100:.1f}% della media: [{r['ci_low']:.6g}, {r['ci_high']:.6g}], IC dim: {dim}")


def main():
    p = argparse.ArgumentParser(description="Calcola intervallo di confidenza della media per una colonna in SQLite.")
    p.add_argument("db", help="Percorso file SQLite (es. LLMtests.db)")
    p.add_argument("--table", default="generated_macm", help="Tabella (default: generated_macm)")
    p.add_argument("--col", default="iterations", help="Colonna numerica (default: iterations)")
    p.add_argument("--conf", type=float, default=0.95, help="Livello confidenza (default: 0.95)")
    p.add_argument("--where", default=None, help="Condizione SQL dopo WHERE (es. 'valid=1')")
    p.add_argument("--limit", default=None, help="Condizione LIMIT dopo WHERE (es. 'limit=5')")
    p.add_argument("--groupby", default=None, help="Colonna per raggruppare (es. 'version' o 'valid')")
    args = p.parse_args()

#    run(args.db, args.table, args.col, args.conf, args.where, args.limit, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 5, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 6, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 7, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 8, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 9, args.groupby)
    run(args.db, args.table, args.col, args.conf, args.where, 10, args.groupby)

if __name__ == "__main__":
    main()