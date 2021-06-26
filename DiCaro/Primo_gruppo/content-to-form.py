# TLN_dicaro_1.5
#  Esperimento content-to-form
#  Usando i dati dell’esercizio 1.1
#  Per ogni concetto, prendere le definizioni a disposizione,
#  Cercare in WordNet il synset corretto
#  Suggerimento: usate il principio del “genus” per indirizzare la ricerca
import time
from pathlib import Path
import pandas as pd
import nltk
from itertools import chain
from collections import Counter
from nltk.corpus import wordnet as wn, stopwords


def words_extraction(sent):
    bag = []
    stop_words = set(stopwords.words('english'))
    punctuation = {',', ';', '(', ')', '{', '}', ':', '?', '!'}
    # Return a tokenized copy of text, using NLTK’s recommended word tokenizer (Treebank + PunkSentence)
    tokens = nltk.word_tokenize(sent)
    for tok in tokens:
        if tok.lower() not in stop_words:
            if tok.lower() not in punctuation:
                bag.append(tok.lower())
    return list(bag)


def compute_genus(concept, num=4):
    extracted_c = concept.apply(lambda x: words_extraction(x.lower())[:num])
    print(extracted_c.head())
    extracted_c = list(chain.from_iterable(extracted_c))

    # Wrap it in a counter
    counter = Counter(extracted_c)

    genus = []
    for key, value in counter.items():
        if value in sorted(counter.values())[-num:]:
            genus.append(key)
    return genus


def get_synset_hyponyms(genus):
    hypon_l = []
    for gen in genus:
        for ss in wn.synsets(str(gen)):
            # print(f'genus.sense: {str(ss.name())}')
            for hypon in ss.hyponyms():  # trovo gli iponimi
                hypon_l.append(hypon)
                # print(f'hypon_l: {str(hypon_l)}')
    return hypon_l


def infer_similiarity(def1, def2, n=4):
    bow_a = list(words_extraction(def1.lower()))[:n]
    bow_b = list(words_extraction(def2.lower()))[:n]
    result = evaluate_performance(bow_a, bow_b)
    return result


def evaluate_performance(bow1, bow2):
    return len(set(bow1) & set(bow2)) * 2 / (len(bow1) + len(bow2))


def content_to_form_result(hypon_list, concept, title, num=4):
    max_sym = 0
    my_synset = ''
    max_list = []

    for hyp in hypon_list:
        # Here I extract the definition
        lex = []
        definition = hyp.definition()
        lex.append(definition)
        # print('definition: ' + str(definition))
        for ex in hyp.examples():
            lex.append(ex)
            # print('example: ' + str(ex))

        for my_def in concept:
            for wn_def in lex:
                my_sym = infer_similiarity(my_def, wn_def, n=num)
                if my_sym > max_sym:
                    max_list.append([max_sym, hyp])
                    max_sym = my_sym
                    my_synset = hyp
    print('max_sym is: ' + str(max_sym))
    print(f'my wn synset inferred for {title} is: {my_synset}')

    print(f'The best word forms for {title} concept')
    for elem in sorted(max_list, key=lambda x: x[0], reverse=True):
        print(f'Score: {elem[0]} for synset: {elem[1]}')


if __name__ == '__main__':
    path = Path('.') / 'input' / '1-1_defs.csv'
    start = time.time()

    df = pd.read_csv(path).dropna()

    courage = df['Courage']
    paper = df['Paper']
    apprehension = df['Apprehension']
    sharpener = df['Sharpener']

    print("I verify the terms exist in WN and have associated synsets")
    term_list = ['courage', 'paper', 'apprehension', 'sharpener']

    for term in term_list:
        print('Term: ' + str(term))
        for ss in wn.synsets(term):
            print(ss)
        print()

    sent = 'Hi, my name is Damiano and I am here to implement and format this code'
    hello = words_extraction(sent)
    print(sent)
    print(f'testing the methodology {hello}\n')

    Genus_Courage = compute_genus(concept=courage, num=3)
    hypon_courage = get_synset_hyponyms(genus=Genus_Courage)

    def1 = 'The ability to do something very hard and constancy'
    def2 = 'The capacity to do something quite hard and with constance'
    test = infer_similiarity('%s' % def1,
                             '%s' % def2)
    print(f'\nAnother test for functionality: similarity score between:\n{def1} \n{def2} is: {test}\n')

    content_to_form_result(hypon_list=hypon_courage, concept=courage, title=term_list[0])

    concept_list_no_courage = [paper, apprehension, sharpener]
    term_list = term_list[1:]

    # how much we want to dig in wordnet
    depth = 6
    for (c, term) in zip(concept_list_no_courage, term_list):
        print(f'\n Computing concept {term}\n')
        genus = compute_genus(concept=c, num=depth)
        hypon = get_synset_hyponyms(genus=genus)
        content_to_form_result(hypon_list=hypon, concept=c, title=term, num=depth)

    end = time.time()
    print('\nTotal time: {} seconds'.format(end - start))



