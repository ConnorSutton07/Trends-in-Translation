from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

def sentiment_score(text: str) -> dict:
    info = SentimentIntensityAnalyzer().polarity_scores(text)
    return info['pos'] - info['neg']

def lemmatize(text: str) -> str:
    return WordNetLemmatizer().lemmatize(text)
 
def sentiment_by_section(text: List[str]) -> List[float]:
    scores = np.zeros((len(text) + 1, ))
    for i, section in enumerate(text):
        scores[i] = sentiment_score(section)
    return scores

def preprocess_text(text: str, lemmatize: bool = True):
    """ 
    Prepares text to be analyzed with word embeddings, sentiment analysis, etc.
    Removes stopwords and unnecessary characters
    Optionally lemmatizes words (ideal for creating word embeddings)

    """
    text = re.sub(r'\W', ' ', str(text)) # remove all the special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # remove all single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) # Remove single characters from the start
    text = re.sub(r'\s+', ' ', text, flags=re.I) # Substituting multiple spaces with single space
    text = text.lower()

    if lemmatize:
        tokens = text.split()
        tokens = [self.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in self.stopwords]
        text = ' '.join(tokens)
    
    return text



    