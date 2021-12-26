import sys
import os 
import json
import argparse
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
from core import analysis
from core import settings
from core import ui
from core.translation import Translation
from adjustText import adjust_text
from tqdm import tqdm

class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]    = os.getcwd()
        self.paths["texts"]      = os.path.join(self.paths["current"], "texts")
        self.paths["Persai"]     = os.path.join(self.paths["texts"], "Persai")
        self.paths["Gallicum"]   = os.path.join(self.paths["texts"], "Gallicum")
        self.paths["figures"]    = os.path.join(self.paths["current"], "figures")
        self.paths["sentiment"]  = os.path.join(self.paths["figures"], "sentiment")
        self.paths["wordclouds"] = os.path.join(self.paths["figures"], "wordclouds")
        self.paths["embeddings"] = os.path.join(self.paths["figures"], "embeddings")

        self.texts = [
            ("Commentarii de Bello Gallico", "Gallicum"),
            ("Persai", "Persai")
        ]

        self.modes = [
            ("Create Embeddings", self.embeddings),
            ("Animate Embeddings", self.animated_embeddings),
            ("Generate Wordclouds", self.wordclouds),
            ("Back", self.run)
        ]

        print()
        print("     +" + "-"*21 + "+")
        print("      TRENDS IN TRANSLATION")
        print("     +" + "-"*21 + "+")
        print()

    def run(self, _ = None, __ = None) -> None:
        #ui.runModes(modes)
        print("here")
        translation_path = self.select_translation()
        if translation_path is None:
            return
        with open(os.path.join(self.paths[translation_path], "info.json")) as infile:
            data = json.load(infile)
        translations = []
        for info in data:
            translations.append(Translation(info, self.paths[translation_path]))
        translations.sort(key = lambda t: t.year)
        self.select_mode()(translations, translation_path)

    def select_mode(self) -> callable:
        num_modes = len(self.modes)
        msg = "Analysis Method:"
        for i, mode in enumerate(self.modes, start=1):
            msg += f"\n   {i}) {mode[0]}"
        index = ui.getValidInput(msg, dtype=int, valid=range(1, num_modes + 1)) - 1
        return self.modes[index][1]

    def select_translation(self) -> str:
        num_texts = len(self.texts)
        msg = "Select Source Text:"
        for i, text in enumerate(self.texts, start=1):
            msg += f"\n   {i}) {text[0]}"
        back_index = num_texts + 1
        msg += f"\n   {back_index}) Exit"
        index = ui.getValidInput(msg, dtype=int, valid=range(1, num_texts + 2)) - 1
        if index != back_index - 1:
            return self.texts[index][1]
        return None
        
    def sentiment(self, translations: list) -> None:
        fig = plt.figure()
        title = "Sentiment Over Time"
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

    def wordclouds(self, translations: list, text_path: str) -> None:
        print("Generating wordclouds for the following translations:")
        for t in translations:
            t.print_info()
            plt.figure()
            plt.imshow(t.generate_wordcloud(stopwords = settings.stopwords, size = (400, 400)), interpolation = 'bilinear', cmap = 'Paired')
            plt.axis('off')
            plt.savefig(os.path.join(self.paths["figures"], text_path, "wordclouds", f"wordcloud_{t.lastname}.jpg"))
            #plt.show()

    def embeddings(self, translations: list, text_path: str, printing: bool = True) -> None:
        print("Creating embeddings...")
        for t in tqdm(translations):
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))

            similar_words, all_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, settings.key_words[text_path])
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
                plt.savefig(os.path.join(self.paths["figures"], text_path, "embeddings", "embeddings_" + t.lastname + ".jpg"), dpi=300)

            if printing:
                print()
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