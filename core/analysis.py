from __future__ import annotations
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from gensim.models.fasttext import FastText
import re 
import numpy as np

def sentiment_score(text: str) -> dict:
    #print(text)
    info = SentimentIntensityAnalyzer().polarity_scores(text)
    return info['pos'] - info['neg']

def lemmatize_text(text: str) -> str:
    return WordNetLemmatizer().lemmatize(text)
 
def sentiment_by_section(text: List[str]) -> List[float]:
    scores = np.zeros((len(text) + 1, ))
    for i, section in enumerate(text):
        scores[i] = sentiment_score(section)
    return scores

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

def preprocess_text(text: List[str], stopwords, lemmatize: bool = True) -> List[str]:
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
            tokens = [lemmatize_text(word) for word in tokens]
            tokens = [word for word in tokens if word not in stopwords]
            block = ' '.join(tokens)
        processed_text.append(block)

    return processed_text

def analyze_embeddings(text):
    embedding_size = 60
    window_size = 40
    min_word = 5
    down_sampling = 1e-2
    #print(text)

    # text = ' '.join(t.preprocess_text()) 
    # final_corpus = [preprocess_text(sentence) for sentence in artificial_intelligence if sentence.strip() !='']

    # word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    # word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

    model = FastText(text,
                    vector_size=embedding_size,
                    window=window_size,
                    min_count=min_word,
                    sample=down_sampling,
                    sg=1) #,
                    # iter=100)

    #print(model.wv['greece'])

    semantically_similar_words = {words: [item[0] for item in model.wv.most_similar([words], topn=5)]
                for words in ['greece', 'persia', 'men', 'women']}

    for k,v in semantically_similar_words.items():
        print(k+":"+str(v))
    print('-----------------------------------')

    