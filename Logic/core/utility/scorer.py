import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.k = 5
        self.b = 0.1
        self.cfs = {}
        self.T = 1000000

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            posting = self.index.get(term, None)
            if posting is None:
                return 0
            idf = np.log10(self.N / len(posting.keys()))
            self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        query_tfs = {}
        for term in query:
            if term in query_tfs.keys():
                query_tfs[term] += 1
            else:
                query_tfs[term] = 1
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        result = {}
        for doc in self.get_list_of_documents(query):
            document_method, query_method = method.split('.')
            score = self.get_vector_space_model_score(query, self.get_query_tfs(query), doc, document_method,
                                                      query_method)
            result[doc] = score
        return result

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        def calculate_w(tf_method, idf_method, tf, idf):
            new_tf = None
            if tf_method == 'n':
                new_tf = tf
            elif tf_method == 'l':
                new_tf = 1 + (np.log10(tf) if tf > 0 else 0)
            new_idf = None
            if idf_method == 'n':
                new_idf = 1
            elif idf_method == 't':
                new_idf = idf
            return new_tf * new_idf

        doc_vec = []
        q_vec = []

        for term, tf in query_tfs.items():
            docs = self.index.get(term, None)
            raw_idf = self.get_idf(term)
            w_q = calculate_w(query_method[0], query_method[1], tf, raw_idf)
            w_d = calculate_w(document_method[0], document_method[1], 0 if docs is None else docs.get(document_id, 0),
                              raw_idf)
            q_vec.append(w_q)
            doc_vec.append(w_d)
        if query_method[2] == 'c':
            q_vec = q_vec / np.linalg.norm(q_vec)
        if document_method[2] == 'c':
            doc_vec = doc_vec / np.linalg.norm(doc_vec)
        return np.dot(q_vec, doc_vec)

    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        result = {}
        for doc in self.get_list_of_documents(query):
            score = self.get_okapi_bm25_score(query, doc, average_document_field_length, document_lengths[doc])
            result[doc] = score
        return result

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_length):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_length : long
            The document's length.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        const = self.k * ((1 - self.b) + (self.b * document_length / average_document_field_length))
        score = 0
        for term in query:
            tf = self.index[term].get(document_id, 0)
            score += self.get_idf(term) * (self.k + 1) * tf / (const + tf)
        return score

    def compute_collection_frequencies(self, term):
        if term not in self.cfs.keys():
            cf = 0
            for doc in self.get_list_of_documents(term):
                cf += self.index[term].get(doc, 0)
            self.cfs[term] = cf

        return self.cfs[term]

    def compute_scores_with_unigram_model(
            self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        result = {}
        for doc in self.get_list_of_documents(query):
            score = self.compute_score_with_unigram_model(query, doc, smoothing_method, document_lengths, alpha, lamda)
            result[doc] = score
        return result

    def compute_score_with_unigram_model(
            self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        score = 1
        if smoothing_method == 'naive':
            for term in query:
                dtf = self.index[term].get(document_id, 0)
                ptmd = dtf / document_lengths[document_id]
                score *= ptmd
        elif smoothing_method == 'bayes':
            for term in query:
                dtf = self.index[term].get(document_id, 0)
                ptmc = self.compute_collection_frequencies(term) / self.T
                ptd = (dtf + alpha * ptmc) / (document_lengths[document_id] + alpha)
                score *= ptd
        elif smoothing_method == 'mixture':
            for term in query:
                dtf = self.index[term].get(document_id, 0)
                ptmc = self.compute_collection_frequencies(term) / self.T
                ptd = lamda * dtf / document_lengths[document_id] + (1 - lamda) * ptmc
                score *= ptd
        return score