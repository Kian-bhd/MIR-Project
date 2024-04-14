import os

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        print(os.getcwd())
        print("meow")
        with open('core/stopwords.txt', 'r') as f:
            self.stopwords = [x for x in f.readlines()]

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        filtered_sentence = []
        for w in query.split():
            if w not in self.stopwords:
                filtered_sentence.append(w)
        return ' '.join(filtered_sentence)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        doc_tokens = doc.split()
        query_tokens = self.remove_stop_words_from_query(query).split()

        for query_word in query_tokens:
            if query_word.casefold() in map(str.casefold, doc_tokens):
                print(query_word)
                indices = [i for i, word in enumerate(doc_tokens) if
                           word.lower() == query_word.lower()]

                for index in indices:
                    start_index = max(0, index - self.number_of_words_on_each_side)
                    end_index = min(len(doc_tokens), index + self.number_of_words_on_each_side + 1)
                    snippet_words = doc_tokens[start_index:end_index]

                    snippet = ' '.join(['***' + word + '***' if word.lower() == query_word.lower() else word for word in
                                        snippet_words])

                    final_snippet += snippet + " ... "

            else:
                not_exist_words.append(query_word)

        print("FINAL")
        print(final_snippet)
        return final_snippet, not_exist_words
