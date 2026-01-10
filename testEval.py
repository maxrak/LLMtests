import pandas as pd

# === 0. Opzioni di stampa: niente '...' ===
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# === 1. Leggi il file CSV ===
input_csv = "dati_processo.csv"
df = pd.read_csv(input_csv)

# === 2. Conversione a datetime ===
df["created_at"] = pd.to_datetime(df["created at"])
df["updated_at"] = pd.to_datetime(df["updated at"])

# === 3. Calcolo tempi ===
df["exec_time_sec"] = (df["updated_at"] - df["created_at"]).dt.total_seconds()
df["exec_time_min"] = df["exec_time_sec"] / 60.0

# === 4. Raggruppamento per combinazione ===
group_cols = ["case", "version", "mode", "RAG", "success"]

grouped = df.groupby(group_cols)["exec_time_sec"]

result = (
    df.groupby(group_cols)
      .agg(
          avg_exec_time_sec=("exec_time_sec", "mean"),
          std_exec_time_sec=("exec_time_sec", "std"),
          min_exec_time_sec=("exec_time_sec", "min"),
          max_exec_time_sec=("exec_time_sec", "max"),
          num_tests=("exec_time_sec", "size")
      )
      .reset_index()
)

# === 5. Coefficiente di variazione per combinazione ===
result["cv"] = result["std_exec_time_sec"] / result["avg_exec_time_sec"]

# === 6. Stampa completa ===
print(result.to_string(index=False))

# === 7. Statistiche globali ===
global_mean = df["exec_time_sec"].mean()
global_std = df["exec_time_sec"].std()
global_min = df["exec_time_sec"].min()
global_max = df["exec_time_sec"].max()
global_cv = global_std / global_mean if global_mean != 0 else float("nan")

print("\n=== STATISTICHE GLOBALI ===")
print(f"Min tempo: {global_min:.2f} s")
print(f"Max tempo: {global_max:.2f} s")
print(f"Media tempo: {global_mean:.2f} s ({global_mean/60:.2f} min)")
print(f"Deviazione standard: {global_std:.2f} s")
print(f"Coefficiente di variazione: {global_cv:.4f} ({global_cv*100:.2f}%)")