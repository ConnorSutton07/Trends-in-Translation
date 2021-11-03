from __future__ import annotations
import nltk
import os
import numpy as np
import re

from typing import List
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


class Translation:
    def __init__(self, info: dict, path: str, delimiter: str = '#') -> Translation:
        self.translator = info['translator']
        self.lastname = self.translator.split(" ")[-1]
        self.year = info['year']
        self.file = info['text-file']
        self.text = self.load_text(os.path.join(path, self.file))
        self.lines = len(self.text)
        self.polarity_score = SentimentIntensityAnalyzer().polarity_scores
        self.lemmatize = WordNetLemmatizer().lemmatize
        self.delimiter = delimiter
        self.stopwords = set(nltk.corpus.stopwords.words('english') + ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0"])

    def print_info(self) -> None:
        print(f"{self.translator}, {self.year}. Lines: {self.lines}")

    def get_info(self) -> str:
        return self.translator + ", " + str(self.year)

    def sentiment_by_line(self) -> np.array:
        sentiment = np.zeros((self.lines,))
        for i, line in enumerate(self.text):
            sentiment[i] = self.polarity_score(line)['compound']
        return sentiment
    
    def sentiment_by_interval(self, divisor: int):
        interval = self.lines // divisor
        if self.lines % divisor != 0: interval += 1
        scores = np.zeros((divisor,))
        index = 0
        for i in range(0, self.lines, interval):
            section = self.text[i : i + interval]
            info = self.polarity_score(" \n ".join(section))
            scores[index] = self.distance_score(info) 
            index += 1
        return scores

    def sentiment_by_section(self, amplification = 1):
        entire_text = " \n ".join(self.text)
        sections = entire_text.split('#')
        for i, section in enumerate(sections):
            sections[i] = self.preprocess_text(sections[i])
        #sections = self.preprocess_text()
        #print(entire_text)
        print(f"Sections: {len(sections)}")
        #
        scores = np.zeros((len(sections) + 1, ))

        index = 0
        for section in sections:
            if index < len(sections) - 1:
                section = section[:-1] # remove the delimiter label
            info = self.polarity_score(section)
            scores[index] = self.distance_score(info) * amplification
            index += 1
        return scores

    def sentiment_by_whole(self) -> dict:
        return self.polarity_score(" \n ".join(self.text))

    def distance_score(self, info: dict) -> float:
        score = info['pos'] - info['neg']
        #return score
        sign = score / np.abs(score) if score != 0 else 1
        return (score ** 2) * sign

    def generate_wordcloud(self, size) -> WordCloud:
        stopwords = set(STOPWORDS)
        stopwords.update(self.stopwords)
        return WordCloud(stopwords = stopwords, background_color = "white", width = size[0], height = size[1]).generate(" \n ".join(self.text))

    @staticmethod
    def load_text(file: str) -> List[str]:
        with open(file, encoding = "utf8") as f:
            contents = f.readlines()
        return contents

    @staticmethod
    def preprocess_text(text: str, lemmatize: bool = True):
        #Remove all the special characters
        text = re.sub(r'\W', ' ', str(text))

        # remove all single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

        # Remove single characters from the start
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

        # Substituting multiple spaces with single space
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        text = text.lower()
        #block = [word for word in block if word not in self.stopwords]
        #print(block)

        if lemmatize:
            tokens = text.split()
            #print(tokens)
            tokens = [self.lemmatize(word) for word in tokens]
            tokens = [word for word in tokens if word not in self.stopwords]
            #tokens = [word for word in tokens if len(word) > 3]
            #print(tokens)
            text = ' '.join(tokens)
        
        return text

        # text = ' '.join(self.text)
        # sections = text.split('#')

        # for i, section in enumerate(sections):
        #     # Remove all the special characters
        #     section = re.sub(r'\W', ' ', str(section))

        #     # remove all single characters
        #     section = re.sub(r'\s+[a-zA-Z]\s+', ' ', section)

        #     # Remove single characters from the start
        #     section = re.sub(r'\^[a-zA-Z]\s+', ' ', section)

        #     # Substituting multiple spaces with single space
        #     section = re.sub(r'\s+', ' ', section, flags=re.I)

        #     section = section.lower()
        #     #block = [word for word in block if word not in self.stopwords]
        #     #print(block)

        #     if lemmatize:
        #         tokens = section.split()
        #         #print(tokens)
        #         tokens = [self.lemmatize(word) for word in tokens]
        #         tokens = [word for word in tokens if word not in self.stopwords]
        #         #tokens = [word for word in tokens if len(word) > 3]
        #         #print(tokens)

        #         section = ' '.join(tokens)

        #     sections[i] = section 

        # return sections
    
