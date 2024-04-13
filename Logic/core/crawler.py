from requests import get
from bs4 import BeautifulSoup
from selenium import webdriver
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json

from Logic.core.preprocess import Preprocessor


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        "User-Agent": "MyCrawler/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = []
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()
        self.data = []

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('../IMDB_crawled.json', 'w', encoding='utf-8') as f:
            json.dump(self.crawled, f, indent=4)
        print('preprocessing...')
        preprocessor = Preprocessor(self.crawled)
        with open('IMDB_crawled_pre_processed.json', 'w+') as f:
            documents = json.dump(preprocessor.preprocess(), f, indent=1)
            f.close()

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = None

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = None

        self.added_ids = None

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        return get(URL, headers=self.headers)

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)
        soup = BeautifulSoup(response.text, 'html5lib')
        print(soup)

        title_link_wrappers = soup.find_all('a', {'class': 'ipc-title-link-wrapper'})
        for title in title_link_wrappers:
            if title['href'].startswith('/title'):
                partial_url = title['href']
                url = 'https://www.imdb.com' + partial_url
                url = '/'.join(url.split('/')[:5])
                self.not_crawled.append(url)
                self.added_ids.append(self.get_id_from_URL(url))

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=25) as executor:
            while crawled_counter < self.crawling_threshold:
                URL = self.not_crawled.pop(0)
                futures.append(executor.submit(self.crawl_page_info, URL))
                if not self.not_crawled:
                    wait(futures)
                else:
                    crawled_counter += 1

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")

        response = self.crawl(URL)
        movie_dict = self.get_imdb_instance()
        self.extract_movie_info(response, movie_dict, URL)
        print(movie_dict)
        self.add_list_lock.acquire()
        self.crawled.append(movie_dict)
        self.add_list_lock.release()

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup = BeautifulSoup(res.text, 'html5lib')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        summary_soup = BeautifulSoup(self.crawl(self.get_summary_link(URL)).text, 'html5lib')
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        review_soup = BeautifulSoup(self.crawl(self.get_review_link(URL)).text, 'html5lib')
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        self: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return url + '/plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return url + '/reviews'
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title = soup.find('span', {'class': 'hero__primary-text'}).text
            return title
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            first_page_summary = soup.find('span', {'role': 'presentation', 'data-testid': 'plot-xs_to_m'}).text
            return first_page_summary
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            director = soup.find('section', {'data-testid': 'title-cast'}).find(string='Director')
            if director is None:
                directors = soup.find('section', {'data-testid': 'title-cast'}).find(string='Directors')
                directors_cum = directors.findNext().get_text('$', strip=True)
                return directors_cum.split("$")
            director = director.findNext()
            return [director.text]
        except:
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars = soup.find_all('a', {'data-testid': 'title-cast-item__actor'})
            return [x.text for x in stars]
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writer = soup.find('section', {'data-testid': 'title-cast'}).find(string='Writer')
            if writer is None:
                writers = soup.find('section', {'data-testid': 'title-cast'}).find(string='Writers')
                writers_cum = writers.findNext().get_text('$', strip=True)
                return writers_cum.split("$")
            writer = writer.findNext()
            return [writer.text]
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related = soup.find_all('div', {
                'class': 'ipc-sub-grid ipc-sub-grid--page-span-2 ipc-sub-grid--nowrap ipc-shoveler__grid'})
            related = related[2].find_all_next('a', {'class': 'ipc-lockup-overlay ipc-focusable'})
            related_links = []
            for title in related:
                if title['href'].startswith('/title'):
                    partial_url = title['href']
                    url = 'https://www.imdb.com' + partial_url
                    url = '/'.join(url.split('/')[:5])
                    self.add_list_lock.acquire()
                    id_ = self.get_id_from_URL(url)
                    related_links.append(url)
                    if id_ in self.added_ids:
                        self.add_list_lock.release()
                        continue
                    self.not_crawled.append(url)
                    self.added_ids.append(self.get_id_from_URL(url))
                    self.add_list_lock.release()
            return related_links
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            summaries = soup.find('div', {'data-testid': 'sub-section-summaries'})
            summaries_cum = summaries.find_all('li', {'data-testid': 'list-item'})
            return [x.text for x in summaries_cum]
        except:
            print("failed to get summary")

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            synopsis = soup.find('div', {'data-testid': 'sub-section-synopsis'})
            return [synopsis.text]
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews_cum = soup.find_all('div', {'class': 'lister-item-content'})
            reviews_list = []
            for review in reviews_cum:
                score = review.find('span', {'class': 'point-scale'})
                if score is not None:
                    score = score.previousSibling.text
                else:
                    score = '-'
                text = review.find('div', {'class': 'text show-more__control'})
                if text is not None:
                    text = text.text
                reviews_list.append([text, score])
            return reviews_list
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            genres = soup.find_all(attrs={'class': 'ipc-chip__text'})
            genres_cum = [x.text for x in genres]
            genres_cum.pop(-1)
            return genres_cum
        except:
            print("Failed to get genres")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            rating = soup.find('span', {'class': 'sc-bde20123-1 cMEQkK'}).text
            return rating
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            mpaa = soup.find_all('a', {'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color'})
            return mpaa[6].text
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            release_year = soup.find_all('a', {'class': 'ipc-link ipc-link--baseAlt ipc-link--inherit-color'})
            return release_year[5].text
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            languages = soup.find('span', string='Languages')
            if languages is None:
                language = soup.find('span', {'class': 'ipc-metadata-list-item__label'}, string='Language').findNext()
                return [language.text]
            languages = languages.findNext()
            languages_cum = languages.get_text('$', strip=True)
            return languages_cum.split('$')
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            countries = soup.find('span', string='Countries of origin')
            if countries is None:
                country = soup.find('span', string='Country of origin').findNext()
                return [country.text]
            countries = countries.findNext()
            countries_cum = countries.get_text('$', strip=True)
            return countries_cum.split('$')
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget = soup.find('li', {'data-testid': 'title-boxoffice-budget'}).text[6:]
            return budget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross = soup.find('li', {'data-testid': 'title-boxoffice-cumulativeworldwidegross'}).text[15:]
            return gross
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=25)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()



if __name__ == '__main__':
    main()
