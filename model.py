#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import logging
import numpy as np
import pandas as pd

import pycaret.nlp
import pycaret.classification
from pycaret.nlp import *
#
logger = logging.getLogger(__name__)
#
class MaliciousWebpageDataset(object):
    """ Class wrapper around the selected dataset. """

    def __init__(self, _file:str, _target_col_topics:str="text_clean", _target_col_class:str="classification", _training_size=0.85, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file, sep=',')

        english_only = True
        min_tokens = 10
        nb_topics = 14
        df = self._df
        # remove rows containing foreign languages
        if english_only and 'is_english' in df:
            df = df[df['is_english'] == True]
        # Keep rows with at least 3 tokens in the `text_clean` column
        df = df[df['nb_tokens'] >= min_tokens ]
        # Remove misclassified rows
        df = df[(df['classification'] == 'benign') | (df['classification'] == 'malicious')]
        # Remove strings containing special characters or
        # misparsed HTML tags and code.
        df['text_clean'] = df['text_clean'].str.replace('_',' ',regex=True)
        df['text_clean'] = df['text_clean'].str.replace('//',' ',regex=True)
        df['text_clean'] = df['text_clean'].str.replace('javascript','',regex=True)
        df['text_clean'] = df['text_clean'].str.replace('https','',regex=True)
        df['text_clean'] = df['text_clean'].str.replace('http','',regex=True)

        print(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        # Add custom stop words here:
        stop_words = ["com", "ca", "go", "td", "tr"
                    "px", "co", "uv", "ru",
                    "mx", "also", "use", 
                    "wo", "may", "oo", "javascript", "www",
                    "html", "id", "class", "http", "https"]

        # Generate the classifier for the text contents of the web pages
        self._clf = pycaret.nlp.setup(data=df, 
                    target=_target_col_topics, 
                    custom_stopwords=stop_words)

        # Create the model
        m_lda = pycaret.nlp.create_model(model='lda', num_topics=nb_topics, multi_core=True)
        d_lda = assign_model(m_lda)
        d_lda.dropna(inplace=True)
        d_lda.drop(['title_raw', 'text_clean', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace=True)

        import matplotlib.pyplot as plt

        # assume "df" is your DataFrame object
        classification_counts = d_lda['classification'].value_counts()
        print(classification_counts)
        # plot the histogram
        classification_counts.plot(kind='bar')

        self._ccls = pycaret.classification.setup(
            data=d_lda,
            transformation=True, 
            normalize=True,
            fix_imbalance=True,
            remove_perfect_collinearity=True,
            target=_target_col_class,
            train_size=_training_size) 

        cl_model = pycaret.classification.create_model('gbc')
        self._best_topics_model = m_lda
        self._best_classifier_model = pycaret.classification.finalize_model(pycaret.classification.tune_model(cl_model))

    @property
    def filename(self) -> str:
        return self._datafile

    @property
    def classifier(self):
        return self._clf

    @property
    def dataframe(self):
        return self._df

    @property
    def nb_rows(self) -> int:
        return self.dataframe.shape[0]

    @property
    def nb_cols(self) -> int:
        return self.dataframe.shape[1]

    @property
    def best_topics_model(self):
        return self._best_topics_model

    @property
    def best_classifier_model(self):
        return self._best_classifier_model

    def save_topics_model_to(self, _filename:str) -> None:
        # Save the best model to a file
        save_model(self._best_topics_model, _filename)

    def save_classifier_model_to(self, _filename:str) -> None:
        # Save the best model to a file
        save_model(self._best_classifier_model, _filename)
