from operator import itemgetter

from graph import LinkGraph
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader
import json
import collections


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.hubs.append(movie['id'])
            self.graph.add_node(movie['id'])
            for star in movie['stars']:
                if star not in self.authorities:
                    self.authorities.append(star)
                if star not in self.graph.nodes:
                    self.graph.add_node(star)
                self.graph.add_edge(star, movie['id'])
        print("INIT DONE")

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            if movie['id'] not in self.graph.nodes:
                self.graph.add_node(movie['id'])
            for star in movie['stars']:
                if star not in self.graph.nodes:
                    self.graph.add_node(star)
                self.graph.add_edge(star, movie['id'])

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = {star: 1 for star in self.authorities}
        h_s = {movie: 1 for movie in self.hubs}

        for iteration in range(5):
            movie_score = 0
            for movie in self.graph.nodes:
                for star in self.authorities:
                    if movie in self.graph.get_successors(star):
                        movie_score += 1
                if movie_score > 0:
                    h_s[movie] = movie_score
            for star in self.graph.nodes:
                star_score = 0
                for movie in self.hubs:
                    if star in self.graph.get_predecessors(movie):
                        star_score += 1
                if star_score > 0:
                    a_s[movie] = star_score
        print(a_s)
        a_s = [(k, v) for k, v in a_s.items()]
        h_s = [(k, v) for k, v in h_s.items()]
        a_s = sorted(a_s, key=itemgetter(1), reverse=True)
        h_s = sorted(h_s, key=itemgetter(1), reverse=True)
        return a_s[:10], h_s[:10]

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    root_set = [{"id": "tt0071562",
        "title": "The Godfather Part II",
        "first_page_summary": "The early life and career of Vito Corleone in 1920s New York City is portrayed, while his son, Michael, expands and tightens his grip on the family crime syndicate.",
        "release_year": "1974",
        "mpaa": "R",
        "budget": "$13,000,000 (estimated)",
        "gross_worldwide": "$47,962,897",
        "rating": "9.0",
        "directors": [
            "Francis Ford Coppola"
        ],
        "writers": [
            "Francis Ford Coppola",
            "Mario Puzo"
        ],
        "stars": [
            "Al Pacino",
            "Robert De Niro",
            "Robert Duvall",
            "Diane Keaton",
            "John Cazale",
            "Talia Shire",
            "Lee Strasberg",
            "Michael V. Gazzo",
            "G.D. Spradlin",
            "Richard Bright",
            "Gastone Moschin",
            "Tom Rosqui",
            "Bruno Kirby",
            "Frank Sivero",
            "Francesca De Sapio",
            "Morgana King",
            "Marianna Hill",
            "Leopoldo Trieste"
        ]}]
    with open('../../IMDB_Crawled.json') as f:
        corpus = json.load(f)
    root_set = corpus[:5]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
