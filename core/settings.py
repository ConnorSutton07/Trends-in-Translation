import nltk

stopwords = set(nltk.corpus.stopwords.words('english') + 
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0"])

replacements = [
    ("men", "man"),
    ("women", "woman"),
    ("greece", "hellas"),
    ("greeks", "hellene"),
    ("greek", "hellene"),
    ("persians", "persian")
]

