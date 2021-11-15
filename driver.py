import sys
import os 
import json
from core.translation import Translation
import core.analysis as analysis
import matplotlib.pyplot as plt 
import numpy as np


class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"] = os.getcwd()
        self.paths["texts"]   = os.path.join(self.paths["current"], "texts")
        self.paths["Persai"]  = os.path.join(self.paths["texts"], "Persai")
        self.paths["figures"] = os.path.join(self.paths["current"], "figures")

    def run(self) -> None:
        with open(os.path.join(self.paths["Persai"], "info.json")) as infile:
            data = json.load(infile)
        translations = []
        for info in data:
            translations.append(Translation(info, self.paths["Persai"]))

        #self.plot_wordcloud(translations)
        #self.plot_sentiment(translations)
        for t in translations:
            t.print_info()
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, t.stopwords)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))
            analysis.analyze_embeddings(corpus)
        
    def plot_sentiment(self, translations: list) -> None:
        fig = plt.figure()
        title = "Aeschylus' Persians | Sentiment Over Time"
        for t in translations:
            t.print_info()
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, t.stopwords)
            #print(text)
            s = analysis.sentiment_by_section(text)
            x = np.arange(s.size)
            plt.plot(x, s, label = t.get_info())    
        plt.title(title)
        plt.legend()
        plt.xticks(np.arange(1, s.size + 1))
        plt.xlabel("Interval")
        plt.ylabel("Positivity / Negativity")
        fig.set_size_inches(12, 6)
        plt.savefig(os.path.join(self.paths["figures"], "persians_comparison_ooo.png"), dpi = 100)
        plt.show()

    def plot_wordcloud(self, translations: list) -> None:
        for t in translations:
            t.print_info()
            plt.figure()
            plt.imshow(t.generate_wordcloud(size = (400, 400)), interpolation = 'bilinear', cmap = 'Paired')
            plt.axis('off')
            plt.savefig(os.path.join(self.paths["figures"], "wordcloud_" + t.lastname + ".jpg"))
            #plt.show()

    

