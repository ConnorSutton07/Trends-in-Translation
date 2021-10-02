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

    def run(self) -> None:
        with open(os.path.join(self.paths["Persai"], "info.json")) as infile:
            data = json.load(infile)

        translations = []
        for info in data:
            translations.append(Translation(info, self.paths["Persai"]))

        for t in translations:
            t.print_info()

        s = translations[0].sentiment_by_line()
        x = np.arange(s.size)

        fig = plt.figure()
        plt.plot(x, s)
        plt.show()