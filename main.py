import retriever as source
from pprint import pprint
import Doe1Factor as doe1
import Doe2Factor as doe2
import DoeTableLatex as doetex
from LLMstartegyEval import CSVgen, analyze_csv

def main():
    rows=source.sessions()
    CSVgen(rows, "dati_processo.csv")
    analyze_csv("dati_processo.csv")
    #apps=doe1.Allresults(data)
    #latex = doetex.stats_to_latex_table(apps, caption="Risultati per app e versione", label="tab:results")
    #print(latex)
    #doe1.printAll(apps)

main()