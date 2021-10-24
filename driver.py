import sys
import os 
import json
from core.translation import Translation
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

        self.plot_wordcloud(translations)
        
    def plot_sentiment(self, translations: list) -> None:
        fig = plt.figure()
        #plt.ylim([-1, 1])
        sections = 30
        title = f"Aeschylus' Persians | Sentiment Over Time"
        for t in translations:
            t.print_info()
            #s = t.sentiment_by_interval(sections)
            s = t.sentiment_by_section()
            x = np.arange(s.size)
            plt.plot(x, s, label = t.get_info())    
        plt.title(title)
        plt.legend()
        plt.xticks(np.arange(1, sections))
        plt.xlabel("Interval")
        plt.ylabel("Positivity / Negativity")
        plt.savefig(os.path.join(self.paths["figures"], "persians_comparison.png"))
        plt.show()

    def plot_wordcloud(self, translations: list) -> None:
        for t in translations:
            t.print_info()
            plt.figure()
            plt.imshow(t.generate_wordcloud(), interpolation = 'bilinear')
            plt.axis('off')
            plt.savefig(os.path.join(self.paths["figures"], "wordcloud_" + t.lastname + ".jpg"))
            #plt.show()