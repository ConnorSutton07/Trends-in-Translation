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
        self.plot_sentiment(translations)
        
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

    def analyze_embeddings(self, t):
        embedding_size = 60
        window_size = 40
        min_word = 5
        down_sampling = 1e-2

        text = ' '.join(t.preprocess_text()) 
        final_corpus = [preprocess_text(sentence) for sentence in artificial_intelligence if sentence.strip() !='']

        word_punctuation_tokenizer = nltk.WordPunctTokenizer()
        word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

        ft_model = FastText(word_tokenized_corpus,
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      iter=100)

        print(ft_model.wv['artificial'])

        semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=5)]
                  for words in ['artificial', 'intelligence', 'machine', 'network', 'recurrent', 'deep']}

        for k,v in semantically_similar_words.items():
            print(k+":"+str(v))