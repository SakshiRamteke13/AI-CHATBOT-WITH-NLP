# preprocess.py - Preprocessing helpers that use spaCy if available, otherwise fallback to NLTK/simple methods.
import re
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
        SPACY_OK = True
    except Exception:
        nlp = None
        SPACY_OK = False
except Exception:
    spacy = None
    nlp = None
    SPACY_OK = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None

class Preprocessor:
    def __init__(self):
        # lazy-download hints handled in README; here we try to set stopwords
        self.stopwords = set()
        self.wnl = None
        if nltk:
            try:
                self.stopwords = set(stopwords.words('english'))
            except Exception:
                self.stopwords = set()
            try:
                self.wnl = WordNetLemmatizer()
            except Exception:
                self.wnl = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        text = self.clean_text(text)
        # If spaCy is available with model, use it
        if SPACY_OK and nlp is not None:
            doc = nlp(text)
            tokens = [t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.is_space)]
            return tokens
        # fallback: use simple split + NLTK lemmatizer if available
        tokens = text.split()
        if self.wnl:
            tokens = [self.wnl.lemmatize(t) for t in tokens if t not in self.stopwords]
        else:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def join(self, text):
        return ' '.join(self.tokenize(text))
