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

        fig = plt.figure()
        #plt.ylim([-1, 1])
        sections = 30
        title = f"Aeschylus' Persians, {sections} sections"
        for t in translations:
            t.print_info()
            s = t.sentiment_by_interval(sections)
            x = np.arange(s.size)
            plt.plot(x, s, label = t.translator)    
        plt.title(title)
        plt.legend()
        plt.xticks(np.arange(sections))
        plt.savefig(os.path.join(self.paths["figures"], "persians_comparison.png"))
        plt.show()
        
            