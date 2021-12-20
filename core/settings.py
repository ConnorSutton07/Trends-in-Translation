import nltk

#-----------PREPROCESSING--------------#

stopwords = set(nltk.corpus.stopwords.words('english') + 
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0", "thine"])

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
TOPN = 2

key_words = {
    'persai': 
       ['greece',
        'greek',
        'persia', 
        'persian', 
        'man', 
        'woman', 
        'woe', 
        'brave',
        'glory'],
    'gallico':
        ['man',
         'woman',
         'victory',
         'brave',
         'glory',
         'rome',
         'romans',
         'caeser',
         'belgae',
         'death',
         'mountain',
         'river']
}

