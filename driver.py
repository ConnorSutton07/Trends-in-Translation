import sys
import os 
import json
import argparse
import core.analysis as analysis
import core.settings as settings
import matplotlib.pyplot as plt 
import numpy as np
from core.translation import Translation


class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]    = os.getcwd()
        self.paths["texts"]      = os.path.join(self.paths["current"], "texts")
        self.paths["Persai"]     = os.path.join(self.paths["texts"], "Persai")
        self.paths["figures"]    = os.path.join(self.paths["current"], "figures")
        self.paths["sentiment"]  = os.path.join(self.paths["figures"], "sentiment")
        self.paths["wordclouds"] = os.path.join(self.paths["figures"], "wordclouds")
        self.paths["embeddings"] = os.path.join(self.paths["figures"], "embeddings")

        self.modes = {
            'sentiment' : self.sentiment,
            'wordclouds': self.wordclouds,
            'embeddings': self.embeddings
        }

        args = self.parse_args()
        self.mode = args.mode

    def run(self) -> None:
        with open(os.path.join(self.paths["Persai"], "info.json")) as infile:
            data = json.load(infile)
        translations = []
        for info in data:
            translations.append(Translation(info, self.paths["Persai"]))
        self.modes[self.mode](translations)
        
    def sentiment(self, translations: list) -> None:
        fig = plt.figure()
        title = "Aeschylus' Persians | Sentiment Over Time"
        for t in translations:
            t.print_info()
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, settings.stopwords)
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
        plt.savefig(os.path.join(self.paths["sentiment"], "sentiment_comparison.png"), dpi = 100)
        plt.show()

    def wordclouds(self, translations: list) -> None:
        for t in translations:
            t.print_info()
            plt.figure()
            plt.imshow(t.generate_wordcloud(stopwords = settings.stopwords, size = (400, 400)), interpolation = 'bilinear', cmap = 'Paired')
            plt.axis('off')
            plt.savefig(os.path.join(self.paths["wordclouds"], "wordcloud_" + t.lastname + ".jpg"))
            #plt.show()

    def embeddings(self, translations: list, printing = False) -> None:
        for t in translations:
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))

            similar_words, all_words, pcs = analysis.analyze_embeddings(corpus, settings.key_words)
            plt.figure(figsize=(18, 10))
            plt.scatter(pcs[:, 0], pcs[:, 1], c='red')
            for word, x, y in zip(all_words, pcs[:, 0], pcs[:, 1]):
                plt.annotate(word, xy=(x+0.01, y+0.0), xytext=(0, 0), textcoords='offset points')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"Word Embeddings for {t.get_info()}") 
            plt.savefig(os.path.join(self.paths["embeddings"], "embeddings_" + t.lastname + ".jpg"))

            if printing:
                t.print_info()
                for k,v in similar_words.items():
                    print(f"{k}: {v}")
                print('-----------------------------------')

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which method of analysis to use')
        args = parser.parse_args()
        return args
