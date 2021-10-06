from __future__ import annotations
from typing import List
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import numpy as np

class Translation:
    def __init__(self, info: dict, path: str) -> Translation:
        self.translator = info['translator']
        self.year = info['year']
        self.file = info['text-file']
        self.text = self.load_text(os.path.join(path, self.file))
        self.lines = len(self.text)
        self.polarity_score = SentimentIntensityAnalyzer().polarity_scores

    def print_info(self) -> None:
        print(f"{self.translator}, {self.year}. Lines: {self.lines}")

    def sentiment_by_line(self) -> np.array:
        sentiment = np.zeros((self.lines,))
        for i, line in enumerate(self.text):
            sentiment[i] = self.polarity_score(line)['compound']
        return sentiment

    def overall_sentiment(self) -> dict:
        return self.polarity_score(" \n ".join(self.text))

    @staticmethod
    def load_text(file: str) -> List[str]:
        with open(file, encoding = "utf8") as f:
            contents = f.readlines()
        return contents