import numpy as np
from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes


class SpellCorrection:
    def __init__(self, all_documents, path='core/indexer/index/'):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.index = {
            idx: Index_reader(path, index_name=idx).index for idx in Indexes
        }
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        if len(word) <= k:
            return shingles.add(word)
        for i in range(len(word) - k + 1):
            shingles.add(word[i:i + k])
        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of json
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        def find_tf(word):
            total = 0
            for idx in [Indexes.SUMMARIES, Indexes.STARS, Indexes.GENRES]:
                try:
                    total += sum(self.index[idx][word].values())
                except:
                    pass
            return total

        for doc in all_documents:
            for field, words in doc.items():
                if field not in [idx.value for idx in Indexes]:
                    continue
                for w in words:
                    if not w.isdigit() and w not in all_shingled_words.keys():
                        all_shingled_words[w] = [self.shingle_word(w, k + 1) for k in range(3)]
                        word_counter[w] = find_tf(w)

        word_counter = dict(sorted(word_counter.items(), key=lambda item: -1 * item[1]))

        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = [(0, None) for _ in range(5)]
        for term, tf in self.word_counter.items():
            word_shingles = self.all_shingled_words[term]
            score = 0
            for k in range(3):
                shingles = self.shingle_word(word, k + 1)
                score += self.jaccard_score(shingles, word_shingles[k])
            score *= np.log10(tf)
            min_idx = top5_candidates.index(min(top5_candidates, key=lambda a:a[0]))
            if score > top5_candidates[min_idx][0]:
                top5_candidates[min_idx] = (score, term)
        top5_candidates = list(sorted(top5_candidates, key=lambda item: item[0], reverse=True))
        return top5_candidates

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = []
        for term in query.split():
            final_result.append(self.find_nearest_words(term)[0][1])
        return ' '.join(final_result)
