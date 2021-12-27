from __future__ import annotations
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def wordcloud(t: Translation, stopwords: List[str], save_path: str) -> None:
    plt.figure()
    plt.imshow(t.generate_wordcloud(stopwords = stopwords, size = (400, 400)), interpolation = 'bilinear', cmap = 'Paired')
    plt.axis('off')
    plt.savefig(save_path)

def embeddings(t: Translation, words: List[str], pcs, explained_variance, save_path: str, adjust_annotations: bool = True) -> None:
    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.family':'serif'})
        plt.scatter(pcs[:, 0], pcs[:, 1], c='darkgoldenrod')

        annotations = []
        for word, x, y in zip(words, pcs[:, 0], pcs[:, 1]):
            annotations.append(plt.annotate(word, xy=(x+0.015, y-0.005), xytext=(0, 0), textcoords='offset points'))
        if adjust_annotations:
            adjust_text(annotations)

        plt.xlabel("PC1 | " + "{:.2%}".format(explained_variance[0]))
        plt.ylabel("PC2 | " + "{:.2%}".format(explained_variance[1]))
        plt.title(f"Translation: {t.get_info()}") 
        plt.savefig(save_path, dpi=200)

def animated_embeddings(embedding_info, save_path: str) -> None:
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
        
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=(len(embedding_info) * N * 2 - N),
        interval = delay,
        fargs = (N, embedding_info),
        repeat = True
    )

    anim.save(save_path)