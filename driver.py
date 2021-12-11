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
            'sentiment' :          self.sentiment,
            'wordclouds':          self.wordclouds,
            'embeddings':          self.embeddings,
            'embeddings-animated': self.animated_embeddings
        }

        args = self.parse_args()
        self.mode = args.mode

    def run(self) -> None:
        with open(os.path.join(self.paths["Persai"], "info.json")) as infile:
            data = json.load(infile)
        translations = []
        for info in data:
            translations.append(Translation(info, self.paths["Persai"]))
        translations.sort(key = lambda t: t.year)
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
            all_words, indices = np.unique(all_words, return_index=True)
            pcs = pcs[indices]

            with plt.style.context('Solarize_Light2'):
                plt.figure(figsize=(12, 8))
                plt.rcParams.update({'font.family':'serif'})
                plt.scatter(pcs[:, 0], pcs[:, 1], c='darkgoldenrod')

                annotations = []
                for word, x, y in zip(all_words, pcs[:, 0], pcs[:, 1]):
                    annotations.append(plt.annotate(word, xy=(x+0.015, y-0.005), xytext=(0, 0), textcoords='offset points'))
                adjust_text(annotations)

                plt.xlabel("PC1 | " + "{:.2%}".format(explained_variance[0]))
                plt.ylabel("PC2 | " + "{:.2%}".format(explained_variance[1]))
                plt.title(f"Translation: {t.get_info()}") 
                plt.savefig(os.path.join(self.paths["embeddings"], "embeddings_" + t.lastname + ".jpg"), dpi=300)

            if printing:
                t.print_info()
                for k,v in similar_words.items():
                    print(f"{k}: {v}")
                print('-----------------------------------')

    def animated_embeddings(self, translations: list, printing = False) -> None:
        embedding_info = []
        for i, t in enumerate(translations):
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements, no_lemmatization = settings.no_lemmatization)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))
            similar_words, all_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, settings.key_words)
            key_words = []
            for k, v in similar_words.items():
                key_words.append(k)
            embedding_info.append([key_words, all_words, pcs, explained_variance, t.get_info()])
        embedding_info.append(embedding_info[0]) # loop back to first translation at the end


        with plt.style.context('Solarize_Light2'):
            fig = plt.figure(figsize=(12, 8))
            ax = plt.axes()
        N = 150
        delay = 20

        def animate(i, N, embedding_info):
            pause = (i // N) % 2 == 0
            ci = i % N 
            j = i // (N * 2)

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
            ax.set_xlim((-0.7, 1.25))
            ax.set_ylim((-0.4, 0.7))
            ax.set_title(embedding_info[j][4])
            #annotations = []
            if pause:
                for word, x, y in zip(embedding_info[j][1],  points[:, 0], points[:, 1]):
                    ax.annotate(word, xy=(x + 0.01, y + 0.0), xytext=(0, 0), textcoords='offset points')
            else:
                for word, x, y in zip(embedding_info[j][1],  points[:, 0], points[:, 1]):
                    if word in embedding_info[j][0]:
                        ax.annotate(word, xy=(x + 0.01, y + 0.0), xytext=(0, 0), textcoords='offset points')

            #adjust_text(annotations)
            
            return ax
            
        anim = animation.FuncAnimation(fig, 
                                       animate, 
                                       frames=(len(embedding_info) * N * 2 - N),
                                       interval = delay,
                                       fargs = (N, embedding_info),
                                       repeat = True)
        anim.save(os.path.join(self.paths["embeddings"], "embeddings_animated.gif"))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices = self.modes.keys(), help = 'determines which method of analysis to use')
        args = parser.parse_args()
        return args
