import time
import os
import json
import copy
from .indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents
        self.path = 'index/'

        self.index = None
        try:
            self.index = self.load_index(self.path)
        except:
            self.index = {
                Indexes.DOCUMENTS.value: self.index_documents(),
                Indexes.STARS.value: self.index_stars(),
                Indexes.GENRES.value: self.index_genres(),
                Indexes.SUMMARIES.value: self.index_summaries(),
            }
            for idx in Indexes:
                self.store_index(self.path, idx)

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id'][0]] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id'][0]
            for star in doc['stars']:
                if star not in current_index.keys():
                    current_index[star] = {}
                if doc_id in current_index[star]:
                    current_index[star][doc_id] = current_index[star][doc_id] + 1
                else:
                    current_index[star][doc_id] = 1
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id'][0]
            for genre in doc['genres']:
                if genre not in current_index.keys():
                    current_index[genre] = {}
                if doc_id in current_index[genre]:
                    current_index[genre][doc_id] = current_index[genre][doc_id] + 1
                else:
                    current_index[genre][doc_id] = 1
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id'][0]
            for term in doc['summaries']:
                if term not in current_index.keys():
                    current_index[term] = {}
                if doc_id in current_index[term]:
                    current_index[term][doc_id] = current_index[term][doc_id] + 1
                else:
                    current_index[term][doc_id] = 1
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return [x for x in self.index[index_type][word]]
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        doc_id = document['id'][0]
        for idx in Indexes:
            if idx == Indexes.DOCUMENTS:
                self.index[idx.value][doc_id] = document
                continue
            for term in document[doc_id]:
                if term not in self.index[idx.value]:
                    self.index[idx.value][term] = {}
                if doc_id in self.index[idx.value][term]:
                    self.index[idx.value][term][doc_id] = self.index[idx.value][term][doc_id] + 1
                else:
                    self.index[idx.value][term][doc_id] = 1
            self.store_index(self.path, idx.value)

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        for idx in Indexes:
            if idx == Indexes.DOCUMENTS:
                self.index[idx.value].pop(document_id)
                continue
            for term, dict_ in self.index[idx.value].items():
                dict_.pop(document_id, -1)
            self.store_index(self.path, idx.value)

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if not index_before_add[index].__contains__(key):
            index_before_add[index].setdefault(key, {})


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return


        self.check_if_key_exists(index_before_add, Indexes.STARS.value, 'tim')

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        self.check_if_key_exists(index_before_add, Indexes.STARS.value, 'henry')

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'drama')

        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'crime')

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        self.check_if_key_exists(index_before_add, Indexes.SUMMARIES.value, 'good')

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'tim')
        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'henry')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'drama')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.SUMMARIES.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + index_name.value + '_index.json', 'w+') as f:
            f.write(json.dumps(self.index[index_name.value], indent=4))
            f.close()

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        current_idx = {}
        for idx in Indexes:
            f = open(path + idx.value + '.json', 'r')
            current_idx[idx.value] = json.load(f)
            f.close()
        return current_idx

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'][0])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# documents = None
# with open('../IMDB_crawled_pre_processed.json', 'r') as f:
#     documents = json.load(f)
#     f.close()
# idx = Index(documents)
# for i in Indexes:
#     assert idx.check_if_index_loaded_correctly(i.value, idx.index[i.value])
# assert idx.check_if_indexing_is_good(Indexes.GENRES.value, 'action')
