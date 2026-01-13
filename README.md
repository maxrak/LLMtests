# LLMtests

Questo repository contiene un tool per l'analisi e la valutazione di strategie di Large Language Models (LLM) utilizzando tecniche di Design of Experiments (DoE). L'obiettivo principale è analizzare l'impatto di diversi fattori sulle prestazioni degli LLM, misurando metriche come successo, numero di iterazioni, validità e consumo di token.

## Obiettivi del Tool

Il tool permette di:
- Recuperare dati da sessioni di test LLM
- Eseguire analisi statistiche attraverso Design of Experiments per identificare fattori significativi che influenzano le prestazioni
- Generare report dettagliati su successo, iterazioni, validità e token utilizzati
- Creare visualizzazioni e tabelle riassuntive per facilitare l'interpretazione dei risultati

## Organizzazione delle Cartelle

- **Root**: Contiene gli script principali Python (`retriever.py`, `doe_analysis.py`, `main.py`, ecc.), file di configurazione JSON e dati CSV.
- **`results/`**: Directory per i risultati delle analisi, inclusi file di testo con output dettagliati e file CSV riassuntivi.
- **`static/`**: Risorse statiche per l'interfaccia web (CSS e JS di Bootstrap).
- **`templates/`**: Template HTML per la visualizzazione dei risultati.
- **`validity/`**: Risultati specifici delle analisi di validità, inclusi file ANOVA, effect sizes e confronti Tukey.
- **`imgs/`**: Immagini generate durante le analisi (grafici, plot).
- **`__pycache__/`**: Cache Python generata automaticamente.

## Utilizzo

Il tool può essere eseguito utilizzando lo script Bash `run.sh`, che automatizza il processo completo:

```bash
./run.sh
```

Questo script:
1. Esegue `retriever.py` per recuperare i dati delle sessioni.
2. Lancia `doe_analysis.py` con diverse configurazioni per analizzare successo, iterazioni, validità e token, salvando i risultati in `results/`.

### Script Principali

- **`retriever.py`**: Recupera dati da sessioni LLM e genera file CSV di input.
- **`doe_analysis.py`**: Esegue l'analisi statistica DoE sui dati CSV, producendo tabelle e statistiche dettagliate.

## File di Configurazione e Dati Disponibili

### Configurazioni (JSON)
- `config_success.json`: Configurazione per l'analisi del successo.
- `config_iter.json`: Configurazione per l'analisi delle iterazioni.
- `config_validity.json`: Configurazione per l'analisi della validità.
- `config_token.json`: Configurazione per l'analisi dei token.

### Dati (CSV)
- `dati_success.csv`: Dati sul successo delle sessioni.
- `dati_iter.csv`: Dati sul numero di iterazioni.
- `dati_validity.csv`: Dati sulla validità.
- `dati_processo.csv`: Dati generali del processo.

Per ulteriori dettagli, consultare i file di configurazione e gli script Python.