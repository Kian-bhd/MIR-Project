import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = []
        doc_list = [x for x in document.split()]
        for i in range(len(doc_list) - (k - 1)):
            shingles.append(' '.join(doc_list[i:i+k]))
        shingles = set(shingles)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shin_docs = [self.shingle_document(doc) for doc in self.documents]
        all_shins = set()
        for s in shin_docs:
            all_shins = all_shins.union(s)
        characteristic_mat = np.zeros((len(all_shins), len(self.documents)), dtype=int)
        for i, shingle in enumerate(all_shins):
            for j, doc in enumerate(shin_docs):
                if shingle in doc:
                    characteristic_mat[i, j] = 1
        return characteristic_mat

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_mat = self.build_characteristic_matrix()
        shin_cnt, doc_cnt = characteristic_mat.shape
        signature_mat = np.full((self.num_hashes, doc_cnt), shin_cnt + 1)
        hash_mat = [np.random.permutation(shin_cnt) for _ in range(self.num_hashes)]
        for i in range(self.num_hashes):
            for j in range(doc_cnt):
                for k in range(shin_cnt):
                    if characteristic_mat[k, j]:
                        signature_mat[i, j] = min(signature_mat[i, j], hash_mat[i][k])
        return signature_mat

    def lsh_buckets(self, signature, bands=None, rows_per_band=None):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        if bands is None:
            if rows_per_band is None:
                bands = 20
                rows_per_band = 5
            else:
                bands = self.num_hashes / rows_per_band
        else:
            if rows_per_band is None:
                rows_per_band = self.num_hashes / bands
            else:
                if bands * rows_per_band != self.num_hashes:
                    raise Exception("Bands and rows per band do not match number of hashes!")

        return

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        return

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
