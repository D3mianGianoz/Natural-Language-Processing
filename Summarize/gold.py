import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(my_corpus, compression_rate):
    """Given a corpus, it learns vocabulary and idf
    Params:
        corpus
    Returns:
        Returns document-term matrix.
    """
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english')
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(my_corpus)
    index = tfidf_vectorizer.get_feature_names()
    values = tfidf_vectorizer_vectors[0].T.todense()
    dataframe = pd.DataFrame(values, index=index, columns=['F-WORD']).sort_values('F-WORD', ascending=False)
    n_selected_values: float = (100 - compression_rate) / 100 * dataframe.shape[0]

    return dataframe.head(int(n_selected_values))