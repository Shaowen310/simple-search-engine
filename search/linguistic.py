import string

from nltk.stem import PorterStemmer

_porter = PorterStemmer()
_punc = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~–—'


def remove_punc(token):
    return token.translate(str.maketrans('', '', _punc))


def remove_digits(token):
    return token.translate(str.maketrans('', '', string.digits))


def lower(token):
    return token.lower()


def stem(token):
    return _porter.stem(token)


def normalize(token):
    t = remove_punc(token)
    t = remove_digits(t)
    if t == '':
        return t
    t = lower(t)
    t = stem(t)
    return t


def generate_normalized_tokens(term_docs):
    for (t, doc) in term_docs:
        t = normalize(t)
        if t == '':
            continue
        yield (t, doc)
