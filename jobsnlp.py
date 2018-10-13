import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import pdftotext
import docx2txt

import os.path


class ResumeJobsRecommender(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''
        Create ResumeJobsRecommender object

        Args:

        Returns:
            None

        '''
        self.tfidf_vect = TfidfVectorizer(tokenizer=self.LemmaTokenizer(),
                                          stop_words=stopwords.words('english'))

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
            recommendations = linear_kernel(self.job_vectors_, resume_vect)
        else:
            recommendations = cosine_similarity(self.job_vectors_, resume_vect)

        if not n_recommendations:
            n_recommendations = self.job_count_

        return list((-recommendations.reshape(-1)).argsort())[:n_recommendations]

    class LemmaTokenizer():
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


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

    with open(pdf_file, "rb") as f:
        pdf = pdftotext.PDF(f)

    return ''.join(pdf)
