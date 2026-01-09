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
    for filename in glob.glob("*.png"):
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

    return {
        "descr_html": descr_html,
        "graphs": graphs,
        "logistic_text": logistic_text,
        "linear_text": linear_text,
        "output_text": analysis_output,
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