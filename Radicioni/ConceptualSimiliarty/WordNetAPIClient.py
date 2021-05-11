from iteration_utilities import deepflatten
from nltk.corpus import wordnet as wn


class WordNetAPIClient:
    """This class implements all the possible operations (API) for accessing to
    WordNet.
    """

    def __init__(self):
        """Constructor. Because computing the max depth of the graph is a very
        expensive task, it is computed here once for all the class.
        """
        print("[Client] - Computing WordNet graph's max depth.")
        self.depth_max = 20
        print("[Client] - WordNet graph's max depth computed.")

    @staticmethod
    def depth_path(synset, lcs):
        """It measures the distance (depth) between the given Synset and the WordNet's root.
        Args:
            synset: synset to reach from the root
            lcs:Lowest Common Subsumer - the first common sense or most specific ancestor node
        Returns:
            the minimum path which contains LCS
        """
        paths = synset.hypernym_paths()
        paths = list(filter(lambda x: lcs in x, paths))  # all path containing LCS
        return min(len(path) for path in paths)

    @staticmethod
    def lowest_common_subsumer(synset1, synset2):
        """
        Args:
            synset1: first synset to take LCS from
            synset2: second synset to take LCS from
        Returns:
            the first common LCS
        """
        if synset2 == synset1:
            return synset2

        commonsArr = []
        for hyper1 in synset1.hypernym_paths():
            for hyper2 in synset2.hypernym_paths():
                zipped = list(zip(hyper1, hyper2))  # merges 2 list in one list of tuples
                common = None
                for i in range(len(zipped)):
                    if zipped[i][0] != zipped[i][1]:
                        break
                    common = (zipped[i][0], i)

                if common is not None and common not in commonsArr:
                    commonsArr.append(common)

        if len(commonsArr) <= 0:
            return None

        commonsArr.sort(key=lambda x: x[1], reverse=True)
        return commonsArr[0][0]

    def distance(self, synset1, synset2):
        """
        Args:
            synset1: first synset to calculate distance
            synset2: second synset to calculate
        Returns:
            distance between the two synset
        """
        lcs = self.lowest_common_subsumer(synset1, synset2)
        if lcs is None:
            return None

        hypernym1 = synset1.hypernym_paths()
        hypernym2 = synset2.hypernym_paths()

        # paths from LCS to root
        hypernym_lcs = lcs.hypernym_paths()

        # create a set of unique items flattening the nested list
        set_lcs = set(deepflatten(hypernym_lcs))

        # remove root
        set_lcs.remove(lcs)

        # path from synset to LCS
        hypernym1 = list(map(lambda x: [y for y in x if y not in set_lcs], hypernym1))
        hypernym2 = list(map(lambda x: [y for y in x if y not in set_lcs], hypernym2))

        # path containing LCS
        hypernym1 = list(filter(lambda x: lcs in x, hypernym1))
        hypernym2 = list(filter(lambda x: lcs in x, hypernym2))

        return min(list(map(lambda x: len(x), hypernym1))) + min(list(map(lambda x: len(x), hypernym2))) - 2

    @staticmethod
    def __depth_max():
        """
        Returns:
            The max depth of WordNet tree (20)
        """
        return max(max(len(path) for path in sense.hypernym_paths()) for sense in wn.all_synsets())

    @staticmethod
    def get_synsets(word):
        """
        Args:
             word: word for which we need to find meaning
        Returns:
            Synset list associated to the given word
        """
        return wn.synsets(word)
