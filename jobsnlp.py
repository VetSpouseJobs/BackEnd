import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


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

        Returns:
            None

        '''
        self.tfidf_vect.fit(jobs)
        self.job_vectors_ = self.tfidf_vect.transform(jobs)
        self.job_count_ = len(jobs)

    def predict(self, resume, n_recommendations=None):
        '''
        Retrun top n_recommendations jobs that match the resume

        Args:
            resume (string): Resume text
            n_recommendations (int): Number of recommendataions to return

        Returns:
            list: indicies of 
        '''
        resume_vect = self.tfidf_vect.transform(resume)
        recommendations = linear_kernel(self.job_vectors_, resume_vect)

        if not n_recommendations:
            n_recommendations = self.job_count_

        return list((-recommendations.reshape(-1)).argsort())[:n_recommendations]

    class LemmaTokenizer():
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
