# TLN_dicaro_2.2

# We download a pre-computed space
import gensim.downloader as gensim_api

# Partendo da un corpus, estrarre i topics con LDA
# - Scelta a mano del numero di topic
# - Provare a ragionare sulla qualità
# Visualizzazione (opzionale) scelta di librerie di visualizzazione di dati testuali

if __name__ == '__main__':
    space = gensim_api.load("glove-wiki-gigaword-300")
