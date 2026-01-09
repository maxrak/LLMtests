"""
Flask web interface for the DOE framework.

This application exposes two main actions via a simple web page:

1. **Scarica dati e genera CSV** — triggers the data download using the
   provided `retriever.py` module. It fetches all experiment records from
   the remote API and writes them into a CSV file (`dati_processo.csv`) via
   the `CSVgen` function. A small helper ensures that a valid certificate
   bundle exists for HTTPS requests by copying the system bundle to
   `public.crt` if necessary. If the API call fails for any reason, the
   user is informed accordingly.

2. **Avvia analisi** — runs the DOE analysis on the generated CSV using
   the `analyze_csv` function from `LLMstrategyEval2.py`. The analysis
   prints extensive textual output and writes a summary table to
   `risultati_DOE_descrittiva.csv`, along with several PNG charts. The
   application captures the printed text, loads the summary table into a
   DataFrame for tabular display, and encodes any generated charts as
   base64 data URIs so they can be embedded directly into the page.

The resulting page presents:

- A descriptive statistics table for every combination of the factors
  (case, version, mode, RAG) in a sortable HTML table.
- The full textual output of the analysis in a preformatted block.
- All generated charts with their filenames as headings.

To run the application locally:

```bash
python3 app.py
```

Then visit http://127.0.0.1:5000 in your browser.  Note that this code
assumes that Flask is installed in the Python environment.  If it is
missing, you can install it with `pip install flask`.
"""

import base64
import glob
import io
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, render_template, request
import re

# Use a non‑interactive Matplotlib backend.  When the DOE analysis
# generates plots within a Flask request handler, the default
# interactive backends (e.g. macOS Cocoa) can cause runtime errors
# because GUI windows may only be created on the main thread.  The
# Agg backend writes figures directly to files without opening any
# windows, eliminating those errors.
import matplotlib
matplotlib.use("Agg")

# Import user‑supplied modules.  These files live in the same directory
# as this script.  They provide the data retrieval and statistical
# analysis functionality.
import retriever
import LLMstrategyEval2


# Resolve the base directory so that the templates folder can be
# referenced correctly regardless of the current working directory.
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# We no longer rely on a global analysis_result.  Each analysis page
# request will compute the results on demand.


def ensure_certificate() -> None:
    """Ensure that a trusted CA bundle is available at ``public.crt``.

    The API endpoint in ``retriever.py`` explicitly passes ``verify="public.crt"``
    to ``requests.get``.  If that file does not exist, requests will
    raise an ``OSError`` complaining about an invalid certificate path.
    To avoid this, copy the system certificate bundle provided by
    ``requests`` into ``public.crt``.  If the file already exists, it
    is left unchanged.
    """
    ca_bundle = requests.certs.where()
    target = Path("public.crt")
    try:
        if not target.exists():
            shutil.copy(ca_bundle, target)
    except Exception:
        # If copying fails we still proceed.  The API call may still
        # succeed if verify is ignored, but retriever.py forces the
        # verify parameter so having a certificate file is important.
        pass


def download_data() -> str:
    """Fetch experiment data via the `retriever` module and write a CSV.

    Returns a human‑readable message describing the outcome.  This
    function catches and returns any exceptions raised during
    retrieval or CSV generation.
    """
    ensure_certificate()
    try:
        # Fetch all experiment sessions from the API.  See retriever.py
        # for implementation details.  The ``sessions`` function
        # internally paginates through the API until all records are
        # collected.
        rows = retriever.sessions()
        if not rows:
            # If no rows were returned it may indicate a connectivity
            # issue.  Still attempt to write the CSV to keep the
            # pipeline consistent.
            message = (
                "Nessun dato trovato. È stato comunque creato un CSV vuoto "
                "per mantenere la struttura di analisi."
            )
        else:
            message = (
                f"Scaricati {len(rows)} record. CSV generato con successo."
            )
        # Write the CSV.  This uses the fieldnames defined in retriever.py
        csv_path = retriever.CSVgen(rows, "dati_processo.csv")
        return f"{message} (file: {csv_path})"
    except Exception as e:
        return f"Errore durante il download dei dati: {e}"


def run_analysis() -> dict:
    """Run the DOE analysis and capture its outputs.

    The analysis is performed on ``dati_processo.csv`` (generated by
    ``download_data``).  The printed output from the analysis is
    captured and returned along with a rendered HTML table of the
    descriptive statistics and any generated charts.  If the CSV is
    missing, an error message is returned instead.
    """
    csv_path = Path("dati_processo.csv")
    if not csv_path.exists():
        return {"error": "Il file CSV dati_processo.csv non esiste. Esegui prima il download."}

    # Redirect stdout to capture the verbose output printed by
    # LLMstrategyEval2.analyze_csv().  Using StringIO allows us to
    # retrieve the entire text afterwards.
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer
    try:
        # Run the analysis.  Setting save_plots=True instructs the
        # function to write PNG charts to the current working
        # directory.  These files will be picked up below and
        # embedded in the HTML.
        LLMstrategyEval2.analyze_csv(str(csv_path), save_plots=True)
    except Exception as exc:
        # Restore stdout before returning.
        sys.stdout = original_stdout
        return {"error": f"Errore durante l'analisi: {exc}"}
    finally:
        # Always restore the original stdout so further logging or
        # errors are not suppressed.
        sys.stdout = original_stdout

    # Retrieve the textual output.
    analysis_output = buffer.getvalue()

    # Load the descriptive results table.  The analysis writes this
    # file under the fixed name ``risultati_DOE_descrittiva.csv``.
    descr_csv = Path("risultati_DOE_descrittiva.csv")
    descr_html = None
    if descr_csv.exists():
        try:
            descr_df = pd.read_csv(descr_csv)
            # Render the DataFrame as HTML.  The classes used here
            # allow some basic styling defined in the template.  The
            # border is removed because our CSS applies its own
            # borders.
            descr_html = descr_df.to_html(index=False, classes="table", border=0)
        except Exception:
            descr_html = None

    # Collect all PNG files created during the analysis.  Each file
    # will be encoded as a base64 string so that it can be embedded
    # directly into the page without writing to a static directory.
    graphs: list[dict] = []
    for filename in glob.glob("imgs/*.png"):
        try:
            with open(filename, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            graphs.append({"name": filename, "data": data})
        except Exception:
            continue

    # --- Parse the analysis output into logistic and linear sections ---
    logistic_text = ""
    linear_text = ""
    lines = analysis_output.splitlines()
    # Find key markers in the text.  Each marker begins at the start of a line.
    try:
        logit_start = next(i for i, l in enumerate(lines) if l.strip().startswith("Generalized Linear Model"))
    except StopIteration:
        logit_start = None
    try:
        linear_start = next(i for i, l in enumerate(lines) if l.strip().startswith(">>> Modello lineare"))
    except StopIteration:
        linear_start = None
    try:
        tukey_start = next(i for i, l in enumerate(lines) if l.strip().startswith(">>> Confronti multipli"))
    except StopIteration:
        tukey_start = None
    # Extract logistic section
    if logit_start is not None:
        # Logistic section ends at the start of the linear section if present
        end = linear_start if linear_start is not None else len(lines)
        logistic_lines = lines[logit_start:end]
        logistic_text = "\n".join(logistic_lines).strip()
    # Extract linear (ANOVA/OLS) section
    if linear_start is not None:
        end = tukey_start if tukey_start is not None else len(lines)
        linear_lines = lines[linear_start:end]
        linear_text = "\n".join(linear_lines).strip()

    # Parse the logistic section into structured tables.  When a
    # logistic section exists, extract summary statistics (model
    # characteristics), parameter estimates and likelihood ratio
    # tests.  Each table is converted to HTML for display in the
    # template.  If parsing fails, the corresponding HTML value
    # remains ``None``.
    logistic_summary_html: str | None = None
    logistic_params_html: str | None = None
    logistic_lr_html: str | None = None
    # Containers for linear model results
    linear_anova_html: str | None = None
    linear_params_html: str | None = None
    linear_stats_html: str | None = None
    tukey_html: str | None = None
    if logistic_text:
        log_lines = logistic_text.splitlines()
        summary_rows: list[dict[str, str]] = []
        header_idx: int | None = None
        lr_section_idx: int | None = None
        # Locate the parameter table header (contains 'coef' and 'std')
        for i, l in enumerate(log_lines):
            if re.search(r"\bcoef\b", l) and re.search(r"std", l):
                header_idx = i
                break
        # Locate the LR test section
        for i, l in enumerate(log_lines):
            if l.strip().startswith("--- Likelihood Ratio test"):
                lr_section_idx = i + 1
                break
        # Build summary rows from lines before the parameter header.  Each
        # line may contain multiple key/value pairs separated by two or
        # more spaces (the statsmodels output aligns values using wide
        # spacing).  Rather than splitting on the first colon, pair
        # adjacent fields: keys end with ':' and the following value
        # holds the entry.  This approach correctly handles values that
        # themselves contain colons (e.g. time strings like '16:30:27').
        if header_idx is not None:
            for line in log_lines[:header_idx]:
                # Skip empty lines or delimiter rows of '=' or '-'
                if not line.strip() or re.match(r"[=\-]{3,}", line.strip()):
                    continue
                # Break the line into segments separated by two or more spaces.
                parts = [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]
                # Iterate over the parts in pairs: the first is the key
                # (ending with a ':'), the second is the value.  If
                # there is an odd number of parts, the last key has no
                # associated value and is ignored.
                for i in range(0, len(parts) - 1, 2):
                    key = parts[i].rstrip(":")
                    value = parts[i + 1]
                    summary_rows.append({"Statistic": key, "Valore": value})
        # Convert summary into HTML
        if summary_rows:
            try:
                summary_df = pd.DataFrame(summary_rows)
                logistic_summary_html = summary_df.to_html(index=False, classes="table table-striped", border=0)
            except Exception:
                logistic_summary_html = None
        # Parse parameter lines
        params_rows: list[dict[str, str]] = []
        if header_idx is not None:
            # Skip header line and following delimiter (two lines)
            start_idx = header_idx + 2
            for line in log_lines[start_idx:]:
                # Stop at delimiter line or LR section
                if not line.strip():
                    continue
                if re.match(r"[=\-]{3,}", line.strip()):
                    continue
                if lr_section_idx is not None and log_lines.index(line) >= lr_section_idx:
                    break
                tokens = line.strip().split()
                # Parameter rows should have at least 7 tokens: term + 6 numeric values
                if len(tokens) < 7:
                    continue
                # Last 6 tokens are numeric; the rest form the term name
                term = " ".join(tokens[:-6])
                coef, std_err, z_val, p_val, ci_low, ci_high = tokens[-6:]
                params_rows.append({
                    "Termine": term,
                    "Coef": coef,
                    "StdErr": std_err,
                    "z": z_val,
                    "p_value": p_val,
                    "CI_inf": ci_low,
                    "CI_sup": ci_high,
                })
        if params_rows:
            try:
                params_df = pd.DataFrame(params_rows)
                logistic_params_html = params_df.to_html(index=False, classes="table table-striped", border=0)
            except Exception:
                logistic_params_html = None
        # Parse likelihood ratio tests
        lr_rows: list[dict[str, str]] = []
        if lr_section_idx is not None:
            for line in log_lines[lr_section_idx:]:
                if not line.strip():
                    continue
                m = re.match(
                    r"Effetto di\s*(.+?):\s*LR=([\d\.eE+\-]+),\s*df=([\d\.eE+\-]+),\s*p=([\d\.eE+\-]+)",
                    line.strip(),
                )
                if m:
                    factor = m.group(1).strip()
                    lr_val = m.group(2).strip()
                    df_val = m.group(3).strip()
                    p_val = m.group(4).strip()
                    lr_rows.append({
                        "Fattore": factor,
                        "LR": lr_val,
                        "df": df_val,
                        "p_value": p_val,
                    })
        if lr_rows:
            try:
                lr_df = pd.DataFrame(lr_rows)
                logistic_lr_html = lr_df.to_html(index=False, classes="table table-striped", border=0)
            except Exception:
                logistic_lr_html = None

    # Parse linear model section into ANOVA and OLS parameter tables
    if linear_text:
        l_lines = linear_text.splitlines()
        # Find the ANOVA section header
        anova_start = None
        for i, l in enumerate(l_lines):
            if l.strip().startswith("--- ANOVA"):
                anova_start = i
                break
        # Find the OLS detail section header
        ols_start = None
        for i, l in enumerate(l_lines):
            if l.strip().startswith("--- Dettaglio modello lineare"):
                ols_start = i
                break
        # Parse ANOVA table
        if anova_start is not None:
            # Header line for ANOVA columns is one line after the section marker
            header_idx = anova_start + 1
            data_idx = anova_start + 2
            anova_cols: list[str] = []
            if header_idx < len(l_lines):
                header_line = l_lines[header_idx].strip()
                # Parse column names separated by whitespace, e.g. 'sum_sq df F PR(>F)'
                anova_cols = [c for c in re.split(r"\s+", header_line) if c]
            anova_rows: list[dict[str, str]] = []
            for line in l_lines[data_idx:]:
                # Stop at empty line, which separates ANOVA from following content
                if not line.strip():
                    break
                if re.match(r"[=\-]{3,}", line.strip()):
                    continue
                tokens = [t for t in re.split(r"\s+", line.strip()) if t]
                if not tokens:
                    continue
                effect = tokens[0]
                values = tokens[1:]
                # Ensure we have at least as many values as columns
                if anova_cols and len(values) < len(anova_cols):
                    continue
                row_dict: dict[str, str] = {"Effetto": effect}
                for col, val in zip(anova_cols, values):
                    row_dict[col] = val
                anova_rows.append(row_dict)
            if anova_rows and anova_cols:
                try:
                    anova_df = pd.DataFrame(anova_rows)
                    # Insert the effect column as first column
                    cols = ["Effetto"] + anova_cols
                    anova_df = anova_df[cols]
                    linear_anova_html = anova_df.to_html(index=False, classes="table table-striped", border=0)
                except Exception:
                    linear_anova_html = None
        # Parse OLS parameter table
        if ols_start is not None:
            # Find header line for parameters (contains 'coef' and 'std') after ols_start
            header_idx = None
            for j in range(ols_start, len(l_lines)):
                if re.search(r"\bcoef\b", l_lines[j]) and re.search(r"std", l_lines[j]):
                    header_idx = j
                    break
            params_rows: list[dict[str, str]] = []
            param_end_idx: int | None = None
            if header_idx is not None:
                # Parameter header columns: we define them explicitly
                param_cols = ["Termine", "Coef", "StdErr", "t", "p_value", "CI_inf", "CI_sup"]
                start_idx = header_idx + 2
                for idx in range(start_idx, len(l_lines)):
                    line = l_lines[idx]
                    stripped = line.strip()
                    # Termination: blank line or a line of '=' characters indicates end of parameter table
                    if not stripped or re.match(r"[=]{3,}", stripped):
                        param_end_idx = idx
                        break
                    if re.match(r"[-]{3,}", stripped):
                        # separator line; continue without updating param_end_idx
                        continue
                    tokens = stripped.split()
                    # Parameter rows should contain at least 7 tokens (term + 6 numeric)
                    if len(tokens) < 7:
                        continue
                    term = " ".join(tokens[:-6])
                    coef, std_err, t_val, p_val, ci_low, ci_high = tokens[-6:]
                    params_rows.append({
                        "Termine": term,
                        "Coef": coef,
                        "StdErr": std_err,
                        "t": t_val,
                        "p_value": p_val,
                        "CI_inf": ci_low,
                        "CI_sup": ci_high,
                    })
                if param_end_idx is None:
                    param_end_idx = len(l_lines)
            if params_rows:
                try:
                    params_df = pd.DataFrame(params_rows)
                    linear_params_html = params_df.to_html(index=False, classes="table table-striped", border=0)
                except Exception:
                    linear_params_html = None
            # Parse model evaluation statistics following the parameter table (e.g., Omnibus, Skew)
            linear_stats_html = None
            if header_idx is not None and 'param_end_idx' in locals():
                # Start parsing after the param_end_idx, skipping delimiter lines
                stats_rows: list[dict[str, str]] = []
                idx = param_end_idx
                # Skip any delimiter lines immediately after the blank line
                while idx < len(l_lines) and (not l_lines[idx].strip() or re.match(r"[=\-]{3,}", l_lines[idx].strip())):
                    idx += 1
                # Parse until blank line or Tukey section
                while idx < len(l_lines):
                    line = l_lines[idx]
                    # Stop at beginning of next section
                    if not line.strip():
                        break
                    if line.strip().startswith('Multiple Comparison') or line.strip().startswith('>>> Confronti multipli'):
                        break
                    # Break the line into parts using two or more spaces as separators
                    parts = [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]
                    # Pair adjacent elements as key-value pairs
                    for i in range(0, len(parts) - 1, 2):
                        key = parts[i].rstrip(':')
                        value = parts[i + 1]
                        stats_rows.append({"Statistic": key, "Valore": value})
                    idx += 1
                if stats_rows:
                    try:
                        stats_df = pd.DataFrame(stats_rows)
                        linear_stats_html = stats_df.to_html(index=False, classes="table table-striped", border=0)
                    except Exception:
                        linear_stats_html = None
            # Parse Tukey HSD multiple comparison results
            tukey_html = None
            # Find the Tukey header
            tukey_header_idx: int | None = None
            for j, line in enumerate(l_lines):
                if 'Multiple Comparison' in line and 'Tukey' in line:
                    tukey_header_idx = j
                    break
            if tukey_header_idx is not None:
                # The actual table header is one line after the header line
                header_line_idx = tukey_header_idx + 2  # skip the line with ======== separators
                # In sample, there is exactly one separator line and header line; adjust if necessary
                # Search forward until we find a header line containing 'group1'
                for k in range(tukey_header_idx, len(l_lines)):
                    if 'group1' in l_lines[k]:
                        header_line_idx = k
                        break
                # Extract column names
                if header_line_idx < len(l_lines):
                    header_parts = [p.strip() for p in re.split(r"\s+", l_lines[header_line_idx].strip()) if p.strip()]
                else:
                    header_parts = []
                # Data lines follow a separator line, so start after header_line_idx+2
                data_start = header_line_idx + 2
                tukey_rows: list[dict[str, str]] = []
                for line in l_lines[data_start:]:
                    if not line.strip():
                        break
                    if re.match(r"[=\-]{3,}", line.strip()):
                        continue
                    tokens = [t.strip() for t in re.split(r"\s+", line.strip()) if t.strip()]
                    # Number of tokens should match header_parts length
                    if not header_parts or len(tokens) != len(header_parts):
                        continue
                    row_dict = {col: val for col, val in zip(header_parts, tokens)}
                    tukey_rows.append(row_dict)
                if tukey_rows and header_parts:
                    try:
                        tukey_df = pd.DataFrame(tukey_rows)
                        tukey_html = tukey_df.to_html(index=False, classes="table table-striped", border=0)
                    except Exception:
                        tukey_html = None

    return {
        "descr_html": descr_html,
        "graphs": graphs,
        "logistic_text": logistic_text,
        "linear_text": linear_text,
        "output_text": analysis_output,
        # New tables for logistic analysis
        "logistic_summary_html": logistic_summary_html,
        "logistic_params_html": logistic_params_html,
        "logistic_lr_html": logistic_lr_html,
        # Tables for linear model (ANOVA and OLS parameters)
        "linear_anova_html": linear_anova_html,
        "linear_params_html": linear_params_html,
        "linear_stats_html": linear_stats_html,
        "tukey_html": tukey_html,
    }


@app.route("/")
def index():
    """Render the landing page with introductory text."""
    return render_template("index.html")


@app.route("/download", methods=["GET", "POST"])
def download_page():
    """
    Page for downloading data from the external API and previewing the CSV.

    On GET, this route displays a brief description and a button to
    initiate the download.  On POST, the data is fetched via
    ``download_data()``, and the resulting CSV is loaded and shown in
    a small preview table.  Any status messages are displayed as
    Bootstrap alerts.
    """
    message = None
    csv_preview_html = None
    if request.method == "POST":
        message = download_data()
        # After downloading, attempt to load the CSV and present a preview.
        csv_path = Path("dati_processo.csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Show only the first 20 rows to avoid overwhelming the page.
                preview_df = df.head(20)
                csv_preview_html = preview_df.to_html(index=False, classes="table table-striped", border=0)
            except Exception as e:
                # If reading fails, we ignore and just show the message.
                csv_preview_html = None
    return render_template(
        "download.html",
        message=message,
        csv_preview_html=csv_preview_html,
    )


@app.route("/analysis")
def analysis_page():
    """
    Page that performs the DOE analysis and displays all results.

    When visited, this route calls ``run_analysis()`` to compute the
    descriptive statistics, logistic regression, linear model results,
    and charts.  The results are then passed into the template for
    rendering.
    """
    result = run_analysis()
    return render_template("analysis.html", result=result)


if __name__ == "__main__":
    # Run the Flask development server.  For production use, a WSGI
    # server such as gunicorn should be used instead.
    app.run(debug=True)