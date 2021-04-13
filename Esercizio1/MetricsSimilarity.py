from math import log


class SimilarityMetrics:
    """
    This class contains the implementations of the three similarity metrics.
    """

    def __init__(self, wordnet_api_client):
        self.client = wordnet_api_client

    def wu_palmer(self, syn1, syn2) -> float:
        """
        Implementations of the Wu-Palmer metric.
        """
        lcs = self.client.lowest_common_subsumer(syn1, syn2)
        if lcs is None:
            return 0

        depth_lcs = self.client.depth_path(lcs, lcs)
        depth_s1 = self.client.depth_path(syn1, lcs)
        depth_s2 = self.client.depth_path(syn2, lcs)

        result = (2 * depth_lcs) / (depth_s1 + depth_s2)
        return result * 1

    def shortest_path(self, syn1, syn2) -> float:
        """
        Implementations of the Shortest Path metric.
        """
        len_s1_s2 = self.client.distance(syn1, syn2)
        if len_s1_s2 is None:
            return 0
        return ((2 * self.client.depth_max - len_s1_s2) / 40) * 1

    def lch(self, syn1, syn2) -> float:
        """
        Implementations of the Leakcock-Chodorow metric.
        """
        len_s1_s2 = self.client.distance(syn1, syn2)
        if len_s1_s2 is None:
            return 0
        return -log(len_s1_s2 + 1 / (2 * self.client.depth_max) + 1) * -10
