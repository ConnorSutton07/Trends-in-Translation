from __future__ import annotations
from typing import List
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import numpy as np

class Translation:
    def __init__(self, info: dict, path: str, delimiter: str = '#') -> Translation:
        self.translator = info['translator']
        self.year = info['year']
        self.file = info['text-file']
        self.text = self.load_text(os.path.join(path, self.file))
        self.lines = len(self.text)
        self.polarity_score = SentimentIntensityAnalyzer().polarity_scores
        self.delimiter = delimiter

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

    def sentiment_by_section(self):
        entire_text = " \n ".join(self.text)
        print(f"Sections: {entire_text.count('#')}")
        sections = entire_text.split('#')
        scores = np.zeros((entire_text.count('#') + 1, ))

        index = 0
        for section in sections:
            info = self.polarity_score(section)
            scores[index] = self.distance_score(info)
            index += 1
        return scores


    def sentiment_by_whole(self) -> dict:
        return self.polarity_score(" \n ".join(self.text))

    def distance_score(self, info: dict) -> float:
        score = info['pos'] - info['neg']
        sign = score / np.abs(score) if score != 0 else 1
        return (score ** 2) * sign


    @staticmethod
    def load_text(file: str) -> List[str]:
        with open(file, encoding = "utf8") as f:
            contents = f.readlines()
        return contents