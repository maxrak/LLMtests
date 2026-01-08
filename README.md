# 📊 LLMtests

![GitHub stars](https://img.shields.io/github/stars/maxrak/LLMtests)
![GitHub forks](https://img.shields.io/github/forks/maxrak/LLMtests)
![GitHub issues](https://img.shields.io/github/issues/maxrak/LLMtests)
![GitHub license](https://img.shields.io/badge/license-Apache%202.0-blue)

**LLMtests** è una raccolta di script e analisi per testare, valutare e visualizzare le prestazioni di diversi modelli o strategie basate su *Large Language Models* (LLM).

## 📁 Struttura del progetto

```
LLMtests/
├── main.py
├── LLMstartegyEval.py
├── retriever.py
├── SummaryOnVersion.py
├── stats.py
├── Doe1Factor.py
├── Doe2Factor.py
├── DoeTableLatex.py
├── dati_processo.csv
├── risultati_stima_successo_iterazioni.csv
├── plot_*.png
├── LLMtests.db
├── sessions.csv
├── ANALYSIS.xlsx
└── ...
```

## 🚀 Introduzione

Questo progetto permette di confrontare strategie, prompt e versioni di LLM, producendo statistiche, database, grafici e report quantitativi.

## 🛠️ Requisiti

- Python ≥ 3.8
- Installazione librerie:

```
pip install -r requirements.txt
```

## ▶️ Utilizzo

### 1. Setup ambiente

```
git clone https://github.com/maxrak/LLMtests.git
cd LLMtests
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Eseguire i test

```
python main.py
```

### 3. Analisi

```
python SummaryOnVersion.py
python stats.py
python Doe1Factor.py
python Doe2Factor.py
```

## 📊 Output

- grafici `.png`
- dataset `.csv`
- analisi `.xlsx`
- archivio `.db`

## 🤝 Contributi

PR, idee e miglioramenti sono benvenuti!

## 📄 Licenza

Distribuito sotto licenza **Apache 2.0**.

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---
