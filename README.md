# Trends-in-Translation

In this project, I use a series of NLP techniques to investigate disparity between translations of the same source text.
The motivation from this project came from noticing contrasting translations from a passage of Marcus Aurelius' *Meditations*:

> "...he will see a kind of bloom and fresh beauty in an old woman or an old man; and he will be able to look with sober eyes on the seductive charm of his own slave boys." 
(Penguin Classics edition; Translator: Martin Hammond, 2006)

> "So will he be able to perceive the proper ripeness and beauty of old age, whether in man or woman: and whatsoever else it is that is beautiful and alluring in whatsoever is, > with chaste and continent eyes he will soon find out and discern." 
(Gutenberg.org version, Translator: Meric Casaubon, 1634)

These two translations diverge greatly in tone, and I thought it might be interesting to attempt to quantitatively measure the degree of difference between a variety of translations of the same ancient source text, and further to search for trends associated with such differences (e.g., year of translation, gender or age of the translator, etc.). There are far more public domain translations of ancient plays than literature, so I chose to analyze seven different translations of Aeschylus' *The Persians*---a Greek tragedy focused on the defeat of Xerxes at Salamis in 480 BCE. 

My primary method of analysis is using word embeddings to spatially compare the contexts of a set of key words across the different translations. Word embeddings are representations of words in vector space such that semantically similar words appear spatially closer to each other than semantically unsimilar words. This allows one to examine what other words are closey associated with a given word; for example, to analyze the portrayal of gender in a given work of literature, a list of the closest semantic neighbors to the words "woman" and "man" could be examined.[^1] 

![semantics](https://user-images.githubusercontent.com/55513603/142302400-518840d1-b170-46c5-9c6e-a19744a6dd2b.png)

Word embeddings are often highly-dimensional, but dimensionality-reduction techniques such as Principal Component Analysis (PCA) can be used to visualize spatial relationships between a set of words.



## References
[1] Pollak, Senja & Martinc, Matej & Poniž, Katja. (2020). Natural Language Processing for Literary Text Analysis: Word-Embeddings-Based Analysis of Zofka Kveder's Work. 


![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Kaulbach%2C_Wilhelm_von_-_Die_Seeschlacht_bei_Salamis_-_1868.JPG/1920px-Kaulbach%2C_Wilhelm_von_-_Die_Seeschlacht_bei_Salamis_-_1868.JPG)


Some initial results, still WIP 

![image](https://user-images.githubusercontent.com/55513603/137642494-70210109-9e35-4d65-8287-ef83189b180b.png)

