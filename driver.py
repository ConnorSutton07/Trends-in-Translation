from __future__ import annotations
import sys
import os 
import json
import argparse
import numpy as np
from core import analysis
from core import settings
from core import ui
from core import graph
from core.translation import Translation
from tqdm import tqdm

class Driver:
    def __init__(self) -> None:
        self.paths = {}
        self.paths["current"]    = os.getcwd()
        self.paths["texts"]      = os.path.join(self.paths["current"], "texts")
        self.paths["Persai"]     = os.path.join(self.paths["texts"], "Persai")
        self.paths["Gallicum"]   = os.path.join(self.paths["texts"], "Gallicum")
        self.paths["Beowulf"]    = os.path.join(self.paths["texts"], "Beowulf")
        self.paths["figures"]    = os.path.join(self.paths["current"], "figures")
        self.paths["sentiment"]  = os.path.join(self.paths["figures"], "sentiment")
        self.paths["wordclouds"] = os.path.join(self.paths["figures"], "wordclouds")
        self.paths["embeddings"] = os.path.join(self.paths["figures"], "embeddings")

        self.texts = [
            ("Commentarii de Bello Gallico", "Gallicum"),
            ("The Persians", "Persai"),
            ("Beowulf", "Beowulf")
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
        while True:
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

    def select_keywords(self) -> List[str]:
        keys = list(settings.key_words.keys())
        num_keys = len(keys)
        msg = "Select Keyword Set:"
        for i, key in enumerate(keys, start=1):
            msg += f"\n   {i}) {key}"
        back_index = num_keys + 1
        msg += f"\n   {back_index}) Back"
        index = ui.getValidInput(msg, dtype=int, valid=range(1, num_keys + 2)) - 1
        if index != back_index - 1:
            return settings.key_words[keys[index]]
        return None

    def wordclouds(self, translations: list, text_path: str) -> None:
        print("Generating wordclouds for the following translations:")
        for t in translations:
            t.print_info()
            save_path = os.path.join(self.paths["figures"], text_path, "wordclouds", f"wordcloud_{t.lastname}.jpg")
            graph.wordcloud(t, settings.stopwords, save_path)

    def embeddings(self, translations: list, text_path: str, printing: bool = True) -> None:
        key_words = self.select_keywords()
        if key_words is None: return
        kwargs = settings.embeddings_kwargs 
        #vector_sizes = [32, 64, 100, 200]
        print("Creating embeddings...")
        for t in tqdm(translations):
            text = t.get_delimited_text()
            counts = np.zeros((len(key_words), ))
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements, no_lemmatization = settings.no_lemmatization)
            corpus = []
            for section in text:
                counts += np.array([section.count(w) for w in key_words])
                corpus.append(section.split(' '))
            table_words, graph_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, key_words, kwargs, settings.neighbors)
            graph_words, indices = np.unique(graph_words, return_index=True)
            pcs = pcs[indices]
            points = analysis.normalize2D(pcs[:, 0], pcs[:, 1])
            plot_save_path = os.path.join(self.paths["figures"], text_path, "embeddings", f"embeddings_{t.lastname}.jpg")
            table_save_path = os.path.join(self.paths["figures"], text_path, "tables", f"table_{t.lastname}.png")
            graph.scatter_embeddings(t, graph_words, points, explained_variance, plot_save_path, adjust_annotations = True)
            graph.tabulate_embeddings(table_words, table_save_path, t.get_info(), settings.neighbors["table"])
            if printing:
                print()
                t.print_info()
                for i, (k,v) in enumerate(table_words.items()):
                    print(f"{k} ({counts[i]}): {v}")
                print('-----------------------------------')

    def animated_embeddings(self, translations: list, text_path: str, printing = False) -> None:
        print("Creating embeddings...")
        embedding_info = []
        for i, t in enumerate(translations):
            text = t.get_delimited_text()
            text = analysis.preprocess_text(text, stopwords = settings.stopwords, replacements = settings.replacements, no_lemmatization = settings.no_lemmatization)
            corpus = []
            for section in text:
                corpus.append(section.split(' '))
            similar_words, all_words, pcs, explained_variance = analysis.analyze_embeddings(corpus, settings.key_words[text_path])
            key_words = []
            for k, v in similar_words.items():
                key_words.append(k)
            embedding_info.append([key_words, all_words, pcs, explained_variance, t.get_info()])
        embedding_info.append(embedding_info[0]) # loop back to first translation at the end
        save_path = save_path = os.path.join(self.paths["figures"], text_path, "embeddings", "embeddings_animated.gif")
        print("Animating embeddings...")
        graph.animated_embeddings(embedding_info, save_path)
