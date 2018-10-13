import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from wordcloud import WordCloud

import pdftotext
import docx2txt

import re
import os.path
import string


class ResumeJobsRecommender(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''
        Create ResumeJobsRecommender object

        Args:

        Returns:
            None

        '''
        self.stop_words = stopwords.words('english') + \
                          list(string.punctuation) + \
                          list(string.whitespace) + \
                          ['\uf0b7']
        self.tfidf_vect = TfidfVectorizer(tokenizer=self.LemmaTokenizer(),
                                          stop_words=stopwords.words('english')+list(string.punctuation),
                                          ngram_range=(1, 3),
                                          strip_accents='unicode')

    def fit(self, jobs):
        '''
        Fit the recommender on all job descriptions in jobs 

        Args:
            jobs (list): list of job descriptions

        Returns:
            None

        '''
        self.tfidf_vect.fit(jobs)
        self.job_vectors_ = self.tfidf_vect.transform(jobs)
        self.vocabulary_ = self.tfidf_vect.vocabulary_
        self.inv_vocabulary_ = {v:k for k, v in self.vocabulary_.items()}
        self.job_count_ = len(jobs)

    def transform(self, jobs):
        '''
        Transform job descriptions in jobs to tf-idf vectors

        Args:
            jobs (list): list of job descriptions

        Returns:
            None

        '''
        return self.tfidf_vect.transform(jobs)

    def predict(self, resume, n_recommendations=None, metric='cosine_similarity'):
        '''
        Return top n_recommendations jobs that match the resume

        Args:
            resume (string): Resume text
            n_recommendations (int): Number of recommendataions to return
            metric (string): "cosine_similarity", "linear_kernel"

        Returns:
            list: indicies of recommended jobs in descending order
        '''
        resume_vect = self.tfidf_vect.transform([resume])

        if metric == 'linear_kernel':
            recommendation_scores = linear_kernel(
                self.job_vectors_, resume_vect)
        else:
            recommendation_scores = cosine_similarity(
                self.job_vectors_, resume_vect)

        if not n_recommendations:
            n_recommendations = self.job_count_

        recommendation_idxs = list(
            (-recommendation_scores.reshape(-1)).argsort())[:n_recommendations]

        return recommendation_idxs, recommendation_scores[recommendation_idxs]

    def generate_wordcloud(self, doc, file_path, max_words=100):
        '''
        Return top n_recommendations jobs that match the resume

        Args:
            doc (string): document from which to generate wordcloud
            file_path (string): .png File path to save image to
            max_words (int): "cosine_similarity", "linear_kernel"

        Returns:
            list: indicies of recommended jobs in descending order
        '''
        wc = WordCloud(background_color="white", random_state=5, max_words=max_words)

        text_vector = np.array(self.tfidf_vect.transform([doc]).todense()).reshape(-1)
        freq_dict = {self.inv_vocabulary_[i]:f for i, f in enumerate(text_vector)}
        wc.generate_from_frequencies(freq_dict)
        wc.to_file(file_path)

    class LemmaTokenizer():
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def read_txt(txt_file):
    '''
    Read txt file to string

    Args:
        txt_file (file): resume txt

    Returns:
        string: resume text
    '''
    if not os.path.isfile(txt_file):
        raise FileNotFoundError(1, f'{txt_file} was not found')

    with open(txt_file, 'r') as f:
        resume = f.read()

    return resume  

def read_docx(docx_file):
    '''
    Convert docx file to string

    Args:
        docx_file (file): resume docx

    Returns:
        string: resume text
    '''
    if not os.path.isfile(docx_file):
        raise FileNotFoundError(1, f'{docx_file} was not found')

    return docx2txt.process(docx_file)

def read_pdf(pdf_file):
    '''
    Convert pdf file to string

    Args:
        pdf_file (file): resume pdf

    Returns:
        string: resume text

    Raises:
        FileNotFoundError

    '''
    if not os.path.isfile(pdf_file):
        raise FileNotFoundError(1, f'{pdf_file} was not found')

    with open(pdf_file, 'rb') as f:
        pdf = pdftotext.PDF(f)

    #filtered_pdf = [w for w in pdf if not w in stopwords]

    #return ''.join(filtered_pdf)
    return ''.join(pdf).replace('\n',' ').replace('\uf0b7', '')
