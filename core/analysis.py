from __future__ import annotations
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from gensim.models.fasttext import FastText
from icecream import ic
from sklearn.decomposition import PCA
import re 
import numpy as np
import matplotlib.pyplot as plt


def sentiment_score(text: str) -> dict:
    info = SentimentIntensityAnalyzer().polarity_scores(text)
    return info['pos'] - info['neg']

def lemmatize_text(text: str) -> str:
    return WordNetLemmatizer().lemmatize(text)
 
def sentiment_by_section(text: List[str]) -> List[float]:
    scores = np.zeros((len(text) + 1, ))
    for i, section in enumerate(text):
        scores[i] = sentiment_score(section)
    return scores

def normalize2D(xs: List[float], ys: List[float]):
    norm = np.zeros( (len(xs), 2) )
    for i, (x, y) in enumerate(zip(xs, ys)):
        x = (x - min(xs)) / (max(xs) - min(xs))
        y = (y - min(ys)) / (max(ys) - min(ys))
        norm[i] = np.array([x, y])
    return norm

def sentiment_by_interval(text: List[str], divisor: int) -> List[float]:
    interval = len(text) // divisor
    if len(text) % divisor != 0: interval += 1
    scores = np.zeros((divisor,))
    index = 0
    for i in range(0, len(text), interval):
        section = text[i : i + interval]
        scores[index] = sentiment_score(" \n ".join(section))
        index += 1
    return scores

def preprocess_text(text: List[str], stopwords = [], lemmatize: bool = True, replacements: List[tuple] = None, no_lemmatization: List[str] = []) -> List[str]:
    """ 
    Prepares text to be analyzed with word embeddings, sentiment analysis, etc.
    Removes stopwords and unnecessary characters
    Optionally lemmatizes words (ideal for creating word embeddings)

    """

    processed_text = []

    for i, block in enumerate(text):
        block = re.sub(r'\W', ' ', str(block)) # remove all the special characters
        block = re.sub(r'\s+[a-zA-Z]\s+', ' ', block) # remove all single characters
        block = re.sub(r'\^[a-zA-Z]\s+', ' ', block) # Remove single characters from the start
        block = re.sub(r'\s+', ' ', block, flags=re.I) # Substituting multiple spaces with single space
        block = block.lower()

        if lemmatize:
            tokens = block.split()
            tokens = [lemmatize_text(word) for word in tokens if word not in no_lemmatization]
            tokens = [word for word in tokens if word not in stopwords]
            block = ' '.join(tokens)

        if replacements != None:
            for u, v in replacements:
                block = block.replace(u, v)

        processed_text.append(block)

    return processed_text

def analyze_embeddings(text: list, key_words: List[str], kwargs: dict, neighbors: dict):
    model = FastText(text, **kwargs)

    table_words = {words: [item[0] for item in model.wv.most_similar([words], topn = neighbors["table"])] for words in key_words}
    graph_words = {words: [item[0] for item in model.wv.most_similar([words], topn = neighbors["graph"])] for words in key_words}
    flattened_graph_words = np.array(sum([[k] + v for k, v in graph_words.items()], []))
    word_vectors = model.wv[flattened_graph_words]
    pca = PCA(n_components = 2)
    p_comps = pca.fit_transform(word_vectors)
    explained_variance = pca.explained_variance_ratio_

    return table_words, flattened_graph_words, p_comps, explained_variance
    