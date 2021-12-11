from __future__ import annotations
import nltk
import os
import numpy as np
import core.settings

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


class Translation:
    def __init__(self, info: dict, path: str, delimiter: str = '#') -> Translation:
        self.translator = info['translator']
        self.lastname = self.translator.split(" ")[-1]
        self.year = int(info['year'])
        self.file = info['text-file']
        self.lines = self.load_text(os.path.join(path, self.file))
        self.text = '\n'.join(self.lines)
        self.delimiter = delimiter

    def print_info(self) -> None:
        print(f"{self.translator}, {self.year}")

    def get_info(self) -> str:
        return self.translator + ", " + str(self.year)

    def get_delimited_text(self) -> List[str]:
        return self.text.split(self.delimiter)
    
    def generate_wordcloud(self, size, stopwords) -> WordCloud:
        return WordCloud(stopwords = stopwords, background_color = "white", width = size[0], height = size[1]).generate(self.text)

    @staticmethod
    def load_text(file: str) -> List[str]:
        with open(file, encoding = "utf8") as f:
            contents = f.readlines()
        return contents


