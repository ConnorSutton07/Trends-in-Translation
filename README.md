# Trends-in-Translation

In this project, I use a series of NLP techniques to investigate disparity between translations of the same source text.
The motivation from this project came from noticing contrasting translations from a passage of Marcus Aurelius' *Meditations*:

> "...he will see a kind of bloom and fresh beauty in an old woman or an old man; and he will be able to look with sober eyes on the seductive charm of his own slave boys." 

(Penguin Classics edition; Translator: Martin Hammond, 2006)

> "So will he be able to perceive the proper ripeness and beauty of old age, whether in man or woman: and whatsoever else it is that is beautiful and alluring in whatsoever is,  with chaste and continent eyes he will soon find out and discern." 

(Gutenberg.org version, Translator: Meric Casaubon, 1634)

These two translations diverge greatly in tone, and I thought it might be interesting to attempt to quantitatively measure the degree of difference between a variety of translations of the same ancient source text, and further to search for trends associated with such differences (e.g., year of translation, gender or age of the translator, etc.). There are far more public domain translations of ancient plays than literature, so I chose to analyze seven different translations of Aeschylus' *The Persians*&mdash;a Greek tragedy focused on the defeat of Xerxes at Salamis in 480 BCE. 

My primary method of analysis is using word embeddings to spatially compare the contexts of a set of key words across the different translations. Word embeddings are representations of words in vector space such that semantically similar words appear spatially closer to each other than semantically unsimilar words. This allows one to examine what other words are closey associated with a given word; for example, consider the list of the closest semantic neighbors to the words "woman" and "man" in the following table which was used to analyze the portrayal of gender in the work of Slovenian author Zofka Kveder.[^1] 

<p align="center">
<img src="https://user-images.githubusercontent.com/55513603/142302400-518840d1-b170-46c5-9c6e-a19744a6dd2b.png" width="700"/>
</p>

Word embeddings are often highly-dimensional, but dimensionality-reduction techniques such as Principal Component Analysis (PCA) can be used to visualize spatial relationships between a set of words:

<p align="center">
<img src="https://github.com/ConnorSutton07/Trends-in-Translation/blob/master/figures/embeddings/embeddings_Theodoridis.jpg" width="700"/>
</p>

Other techniques were also explored. Another means of visually comparing different translations of the same source involves generating wordclouds which emphasize the most common topics in each translation. This would allow one to see some differences, but the method is not quantitative or indicative of anything meaningful&mdash;it can, however, provide hints as to which words may be especially insightful to analyze with word embeddings. 

<p align="center">
<img src="https://user-images.githubusercontent.com/55513603/142324826-12e89e3b-1225-4bc4-8476-8dfa5a7baa1a.png" width="500"/>
</p>


Using sentiment analysis to observe the positivity/negativity of a text over time was also attempted with decent but overall weak results. While visually interesting results can be produced, most sentiment analysis models are trained on data from social media or product reviews and perform poorly in the realm of literature. While the results may be moderately insightful, other techniques such as comparing word embeddings are more robust as models trained on external data are not necessary. 

<p align="center">
<img src="https://user-images.githubusercontent.com/55513603/142321611-d7ab33ce-0498-44bf-9ef7-75828e506b67.png" width="700"/>
</p>




## References

[^1]: Pollak, Senja & Martinc, Matej & Poniž, Katja. (2020). Natural Language Processing for Literary Text Analysis: Word-Embeddings-Based Analysis of Zofka Kveder's Work. <p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Kaulbach%2C_Wilhelm_von_-_Die_Seeschlacht_bei_Salamis_-_1868.JPG/1920px-Kaulbach%2C_Wilhelm_von_-_Die_Seeschlacht_bei_Salamis_-_1868.JPG" width="700"/>
</p>





