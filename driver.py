import sys
import os 
import json
import argparse
import core.analysis as analysis
import core.settings as settings
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
from core.translation import Translation
from adjustText import adjust_text

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

            similar_words, all_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, settings.key_words)

            with plt.style.context('Solarize_Light2'):
                plt.figure(figsize=(12, 8))
                plt.scatter(pcs[:, 0], pcs[:, 1], c='red')

                annotations = []
                for word, x, y in zip(all_words, pcs[:, 0], pcs[:, 1]):
                    annotations.append(plt.annotate(word, xy=(x+0.01, y+0.0), xytext=(0, 0), textcoords='offset points'))
                adjust_text(annotations)

                plt.xlabel(f"PC1 | {explained_variance[0]}")
                plt.ylabel(f"PC2 | {explained_variance[1]}")
                plt.title(f"Translation: {t.get_info()}") 
                plt.savefig(os.path.join(self.paths["embeddings"], "embeddings_" + t.lastname + ".jpg"))

            if printing:
                t.print_info()
                for k,v in similar_words.items():
                    print(f"{k}: {v}")
                print('-----------------------------------')

    def animated_embeddings(self, translations: list, printing = False) -> None:
        embedding_info = []
        for i, t in enumerate(translations):
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))
            similar_words, all_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, settings.key_words)
            embedding_info.append([similar_words, all_words, pcs, explained_variance, t.get_info()])
        embedding_info.append(embedding_info[0])
        
        j = 0 
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes()
        N = 200
        delay = 20

        def animate(i, N, embedding_info):
            pause = (i // N) % 2 == 0
            ci = i % N 
            j = i // (N * 2)
            
            #print(i, j, pause)

            current_points = np.array(list(zip(embedding_info[j][2][:, 0], embedding_info[j][2][:, 1])))
            next_points = current_points if pause else np.array(list(zip(embedding_info[j + 1][2][:, 0], embedding_info[j + 1][2][:, 1])))

            ax.clear()
            points = np.empty((current_points.shape))
            for k in range(len(current_points)):
                #print(current_points[k])
                xdist = next_points[k][0] - current_points[k][0]
                ydist = next_points[k][1] - current_points[k][1]
                word  = embedding_info[j][1][k]
                points[k][0] = current_points[k][0] + ((ci * xdist) / N)
                points[k][1] = current_points[k][1] + ((ci * ydist) / N)
            ax.scatter(points[:, 0], points[:, 1])

            annotations = []
            for word, x, y in zip(embedding_info[j][1],  points[:, 0], points[:, 1]):
                annotations.append(ax.annotate(word, xy=(x + 0.01, y + 0.0), xytext=(0, 0), textcoords='offset points'))
            adjust_text(annotations)
            ax.set_xlim((-0.7, 1.25))
            ax.set_ylim((-0.5, 0.6))
            ax.set_title(embedding_info[j][4])
            return ax
            
        anim = animation.FuncAnimation(fig, 
                                       animate, 
                                       frames=(len(embedding_info) * N * 2),
                                       interval = delay,
                                       fargs = (N, embedding_info))
        anim.save(os.path.join(self.paths["embeddings"], "animated_embeddings.gif"))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which method of analysis to use')
        args = parser.parse_args()
        return args
