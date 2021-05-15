# TLN_dicaro_1.5
#  Esperimento content-to-form
#  Usando i dati dell’esercizio 1.1
#  Per ogni concetto, prendere le definizioni a disposizione,
#  Cercare in WordNet il synset corretto
#  Suggerimento: usate il principio del “genus” per indirizzare la ricerca
from pathlib import Path

from DiCaro.Primo_gruppo.aggregate_concepts import load_data

if __name__ == '__main__':

    path = Path('.') / 'input' / '1-1_defs.csv'
    defs = load_data(path)  # Loading the definition file

    # for d in defs:
    pass
