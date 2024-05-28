import json
import re

import pandas as pd
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if punctuation_removal:
        text = re.sub(r'[^\w\s]', '', text)
    text = text.split()
    text = [token for token in text if len(token) >= minimum_length]
    if stopword_removal:
        stop_words = set(stopwords.words('english')) | set(stopwords_domain)
        text = [token for token in text if token not in stop_words]
    if lower_case:
        text = [token.lower() for token in text]
    return ' '.join(text)


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        records = []
        for dic in data:
            record = {
                "id": dic.get("id"),
                "title": dic.get("title"),
                "summaries": dic.get("summaries", []),
                "genres": dic.get("genres", []),
                "synopsis": dic.get("synopsis", []),
                "reviews": dic.get("reviews", [])
            }
            records.append(record)
        df = pd.DataFrame(records)
        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        df['processed_text'] = df['summaries'].apply(lambda x: preprocess_text(' '.join(x)) if x else "")
        df['genre'] = df['genres'].apply(
            lambda x: x[0] if x else "N/A")
        X = df['processed_text'].values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['genre'])

        return X, y
