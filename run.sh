#!/bin/sh
echo "Retrieving Data"
python retriever.py
echo "DoE on Probability of Success"
python doe_analysis.py configs/config_success.json > results/success.txt
echo "DoE on Token Consumption"
python doe_analysis.py configs/config_success.json > results/token.txt
echo "DoE on Number of Iterations"
python doe_analysis.py configs/config_iter.json > results/iterations.txt
echo "DoE on Model Validity"
python doe_analysis.py configs/config_validity.json >> results/validity.txt
python ./statistiche_edit_distance.py ./dati/dati_validity.csv --confidence 0.90 >results/validity_stats.txt