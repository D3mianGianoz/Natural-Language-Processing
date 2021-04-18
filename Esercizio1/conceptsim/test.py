from MetricsSimilarity import *
from WordNetAPIClient import *

if __name__ == "__main__":
    Api = WordNetAPIClient()

    dog = wn.synset('dog.n.01')
    cat = wn.synset('cat.n.01')
    hit = wn.synset('hit.v.01')
    slap = wn.synset('slap.v.01')
    # wow = Api.lowest_common_subsumer(synset1=cat, synset2=cat)
    # test = Api.distance(synset1=dog, synset2=dog)

    metrics = SimilarityMetrics(Api)

    # Funziona parzialmente
    test1a = metrics.shortest_path(cat, cat)
    # mi da un valore troppo grande
    test1b = metrics.shortest_path(cat, dog)
    test1c = metrics.shortest_path(hit, slap)

    # TODO capire come gestire il -log
    test2a = metrics.lch(cat, cat)
    test2b = metrics.lch(hit, slap)
    test2c = metrics.lch(cat, dog)

    # Corretto
    test3a = metrics.wu_palmer(cat, dog)

    # Esempi presi da https://www.nltk.org/howto/wordnet.html
    # print(f'PHT_sim({s1_c}{s1_d}): {s1_c.path_similarity(s1_d)}')

    print(test1a)
    print(wn.path_similarity(cat, cat, simulate_root=False))
    print("-----------------------------------------")
    print(test1b)
    print(wn.path_similarity(cat, dog, simulate_root=False))
    print("-----------------------------------------")
    print(test1c)
    print(wn.path_similarity(hit, slap, simulate_root=False))
    print("=========================================")

    print(test2a)
    print(wn.lch_similarity(cat, cat))
    print("-----------------------------------------")
    print(test2b)
    print(wn.lch_similarity(hit, slap))
    print("-----------------------------------------")
    print(test2c)
    print(wn.lch_similarity(cat, dog))
    print("=========================================")

    print(test3a)
    print(wn.wup_similarity(cat, dog))
    print("=========================================")


