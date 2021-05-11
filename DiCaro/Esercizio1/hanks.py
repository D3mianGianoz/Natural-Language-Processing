# TLN_dicaro_1.4
import nltk
from nltk.corpus import wordnet
#  Lab (su teoria di P. Hanks)
#  1. Scegliere un verbo transitivo (almeno 2 argomenti)
#  2. Recuperare da un corpus n (> 1000) istanze in cui esso viene usato
#  3. Effettuare parsing e disambiguazione
#  4. Usare i super sensi di WordNet sugli argomenti (subj e obj nel caso di 2
#     argomenti) del verbo scelto
#  5. Aggregare i risultati, calcolare le frequenze, stampare i
#     cluster semantici ottenuti


def filtra(lista: list) -> bool:
    for (word, PS) in lista:
        if word in verb_of_interest:
            if PS == 'VB' or PS == 'VBZ' or PS == 'VBG':
                return True
            else:
                print(PS)
    return False


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


if __name__ == '__main__':
    # 1 I choose said, that have a good context in brown
    verb_of_interest = ["saying", "say", "says"]

    # 2. let's get a list of medium-frequency content words
    nltk.download('brown')

    # already pos tagged sentence
    tagged_sent_of_interest = [ps for ps in nltk.corpus.brown.tagged_sents()]
    # filter with filtra function
    ps = list(filter(filtra, tagged_sent_of_interest))

    sent0 = ps[0]

    # stop word removal
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = {',', ';', '(', ')', '{', '}', ':', '?', '!', '.'}

    # pre process: Lemmatization and
    wnl = nltk.WordNetLemmatizer()
    lemmas = []
    for (elem, POS) in sent0:
        elem = str(elem).lower()
        if elem not in stop_words and elem not in punctuation:
            lemma = wnl.lemmatize(elem, get_wordnet_pos(POS))
            lemmas.append((lemma, POS))





