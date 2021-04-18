from math import log10


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
        return result * 10

    def shortest_path(self, syn1, syn2) -> float:
        """ Implementations of the Shortest Path metric.
        with normalization from [0, 2*max_val] to [0,1]
        new_val = (val - lower_bound)/(upper_bound - lower_bound)
        """
        len_s1_s2 = self.client.distance(syn1, syn2)
        if len_s1_s2 is None:
            return 0
        return ((2 * self.client.depth_max - len_s1_s2) / 40) * 10

    def lch(self, syn1, syn2) -> float:
        """
        Implementations of the Leakcock-Chodorow metric.
        """
        max_val = self.client.depth_max
        len_s1_s2 = self.client.distance(syn1, syn2)
        if len_s1_s2 is None:
            return 0

        if len_s1_s2 == 0:
            sim = log10((len_s1_s2 + 1) / (2 * max_val + 1))
        else:
            sim = log10(len_s1_s2 / (2 * max_val))

        return (- sim / (log10(2 * max_val + 1))) * 10

    def get_all(self):
        """
        Returns:
            a list of reference to the metrics implementation inside this class.
        """
        return [(self.wu_palmer, "Wu & Palmer"), (self.shortest_path, "Shortest Path"),
                (self.lch, "Leakcock & Chodorow")]
