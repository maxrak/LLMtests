#!/bin/sh
python retriever.py
python doe_analysis.py config_success.json > results/success.txt
python doe_analysis.py config_iter.json > results/iterations.txt
python doe_analysis.py config_validity.json >> results/validity.txt
