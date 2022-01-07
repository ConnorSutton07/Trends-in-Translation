import nltk

#-----------PREPROCESSING--------------#

stopwords = set(nltk.corpus.stopwords.words('english') + 
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0", "thine", "chapter"])

no_lemmatization = [
    'salamis',
    'less',
    'was'
]

replacements = [
    ("men", "man"),
    ("women", "woman"),
    ("lady", "woman"),
    ("greatest", "great"),
    ("greece", "hellas"),
    ("greeks", "hellas"),
    ("greek", "hellas"),
    ("persians", "persia"),
    ("persian", "persia")
    # ("helvetian", "helvetii"),
    # ("aeduan", "aedui"),
    # ("roman", "rome"),
    # ("gallic", "gaul"),
    # ("divitiacus", "diviciacus")
]

#----------WORD EMBEDDINGS-------------#
TOPN = 4

embeddings_kwargs = {
    "vector_size": 64,
    "window": 10,
    "min_count": 5,
    "sample": 1e-2,
    "sg": 1
}

key_words = {
    'Persai': [
        'hellas',
        'persia', 
        'man', 
        'woman', 
        'god', 
        'brave',
        'king',
        'xerxes'
    ],
    'Gallicum': [
        'man',
        'woman',
        'caesar',
        'brave',
        'soldier',
        'camp',
        'gaul',
        'rome',
        'senate',
        'enemy',
        'army',
        'great', 
        'war'
    ],
    'Roman & Gallic States and Characters': [
        'caesar',
        'gaul',
        'vercingetorix',
        'aedui',
        'diviciacus',
        'helvetii',
        'german',
        'sequani',
        'ariovistus',
        'bituriges',
        'belgae'
    ],
    'Beowulf': [
        'battle',
        'warrior',
        'king',
        'beowulf',
        'war',
        'man',
        'woman',
        'sword',
        'grendel',
        'hero',
        'good',
        'evil'
    ]
}

