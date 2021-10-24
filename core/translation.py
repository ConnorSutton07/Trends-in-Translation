from __future__ import annotations
from typing import List
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
import numpy as np

class Translation:
    def __init__(self, info: dict, path: str, delimiter: str = '#') -> Translation:
        self.translator = info['translator']
        self.lastname = self.translator.split(" ")[-1]
        self.year = info['year']
        self.file = info['text-file']
        self.text = self.load_text(os.path.join(path, self.file))
        self.lines = len(self.text)
        self.polarity_score = SentimentIntensityAnalyzer().polarity_scores
        self.delimiter = delimiter
        self.stopwords = ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0"]

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

    def sentiment_by_section(self, amplification = 5):
        entire_text = " \n ".join(self.text)
        print(f"Sections: {entire_text.count('#')}")
        sections = entire_text.split('#')
        scores = np.zeros((entire_text.count('#') + 1, ))

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