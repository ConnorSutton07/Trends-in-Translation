import nltk

#-----------PREPROCESSING--------------#

stopwords = set(nltk.corpus.stopwords.words('english') + 
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0", "thine", "chapter"])

no_lemmatization = [
    'salamis'
]

replacements = [
    ("men", "man"),
    ("women", "woman"),
    ("greece", "hellas"),
    ("greeks", "hellene"),
    ("greek", "hellene"),
    ("persians", "persian")
]

#----------WORD EMBEDDINGS-------------#
TOPN = 4

key_words = {
    'Persai': 
       ['greece',
        'greek',
        'persia', 
        'persian', 
        'man', 
        'woman', 
        'woe', 
        'brave',
        'glory'],
    'Gallicum':
        ['man',
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
         'war']
}

