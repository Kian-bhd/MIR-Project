import json
import string
import re
from indexer import index

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        stopwords_file = open("stopwords.txt", "r")
        self.stopwords = [x for x in stopwords_file.readlines()]
        stopwords_file.close()
        self.WNL = WordNetLemmatizer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[json]
            The preprocessed documents.
        """
        for doc in self.documents:
            review_string = []
            for review in doc['reviews']:
                review_string.append(review[0])
                review_string.append(review[1])
            doc['reviews'] = ' '.join(review_string)
            for key in doc.keys():
                if isinstance(doc[key], list):
                    doc[key] = ' '.join(doc[key])
                doc[key] = self.normalize(doc[key])
                doc[key] = self.remove_links(doc[key])
                doc[key] = self.remove_punctuations(doc[key])
                doc[key] = self.remove_stopwords(doc[key])
        return self.documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        norm_str = []
        for x in text.split():
            lwr_x = x.lower()
            lemm_x = self.WNL.lemmatize(lwr_x, pos="v")
            norm_str.append(lemm_x)
        return ' '.join(norm_str)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        regexp = re.compile('|'.join(patterns))
        rem_str = []
        for x in text.split():
            if regexp.match(x):
                continue
            rem_str.append(x)
        return ' '.join(rem_str)

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        filtered_sentence = []
        for w in self.tokenize(text):
            if w not in self.stopwords:
                filtered_sentence.append(w)
        return filtered_sentence


docs = []
with open("../IMDB_Crawled.json", "r") as f:
    docs = json.load(f)
    f.close()
docs = Preprocessor(docs).preprocess()
ind = index.Index(docs)
