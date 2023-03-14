#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pycaret.classification import *
from sklearn.feature_extraction.text import CountVectorizer
#
logger = logging.getLogger(__name__)
#
class MaliciousWebpageDataset(object):
    """ Class wrapper around the selected dataset. """

    def __init__(self, _file:str, _target_col_topics:str="text_clean", _target_col_class:str="classification", _training_size=0.85, _drop_cols=[]) -> None:
        self._datafile = _file
        df_features = pd.read_csv(_file, sep=',')

        # Options
        min_tokens = 10 #@param { type:"integer" }
        max_tokens = 50 #@param { type:"integer" }
        max_words = 250 #@param { type:"integer" }
        english_only = True #@param { type:"boolean" }

        # remove rows containing foreign languages
        if english_only and 'is_english' in df_features:
            df_features = df_features[df_features['is_english'] == True]
        # Drop unneeded columns
        df_features.drop('is_english', axis=1, inplace=True)
        df_features.drop('title_raw', axis=1, inplace=True)
        # Keep rows with at least 3 tokens in the `text_clean` column
        df_features = df_features[df_features['nb_tokens'] >= min_tokens ]
        # Remove misclassified rows
        df_features = df_features[(df_features['classification'] == 'benign') | (df_features['classification'] == 'malicious')]
        # Remove strings containing special characters or
        # misparsed HTML tags and code.
        df_features['text_clean'] = df_features['text_clean'].str.replace('_',' ',regex=True)
        df_features['text_clean'] = df_features['text_clean'].str.replace('//',' ',regex=True)
        df_features['text_clean'] = df_features['text_clean'].str.replace('javascript','',regex=True)
        df_features['text_clean'] = df_features['text_clean'].str.replace('https','',regex=True)
        df_features['text_clean'] = df_features['text_clean'].str.replace('http','',regex=True)

        print(f"The dataset contains {df_features.shape[0]} rows and {df_features.shape[1]} columns.")
        # Load the default list of English stop words
        default_stop_words = stopwords.words('english')

        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        # Add custom stop words here:
        custom_stop_words = ["com", "ca", "go", "td", "tr"
                    "px", "co", "uv", "ru",
                    "mx", "also", "use", 
                    "wo", "may", "oo", "javascript", "www",
                    "html", "id", "class", "http", "https"]

        # Append the custom list to the default list of stop words
        stop_words = default_stop_words + custom_stop_words

        # Define a custom token pattern that matches only alphabetic characters
        pattern = r'\b[A-Za-z]+\b'

        # Create a CountVectorizer
        # We keep only bigrams and trigrams
        # We remove words not withing the [.015, 0.8] frequency
        self._count_vectorizer = CountVectorizer(min_df=.015, 
                                        max_df=0.8, 
                                        stop_words=stop_words, 
                                        max_features=max_words, 
                                        ngram_range=[2, 3],
                                        token_pattern=pattern)

        # Fit the vectorizer to the text data and transform the data
        X = self._count_vectorizer.fit_transform(df_features['text_clean'])

        # Create a DataFrame from the `csr_matrix` generated
        df_words = pd.DataFrame(data=X.toarray(),
                                columns=self._count_vectorizer.get_feature_names())

        # Reset the index of the two DataFrames
        df_features.reset_index(drop=True, inplace=True)
        df_words.reset_index(drop=True, inplace=True)
        # Concatenate the 2 `DataFrame` to generate the dataset
        df = pd.concat([df_features, df_words], axis=1)

        #@title Classifier Options
        sid = 1337 #@param {type:"integer"}
        training_size = 0.85 #@param { type:"number" }

        # Create a PyCaret Classification experiment
        clf = setup(data=df,
                    session_id=sid,
                    transformation=True, 
                    normalize=True,
                    fix_imbalance=True,
                    remove_perfect_collinearity=True,
                    train_size=training_size,
                    target="classification")

        # Compare multiple models and select the best
        best_model = compare_models()

        # Finalize the best model
        final_model = finalize_model(tune_model(best_model))
        self._best_model = final_model

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
    def best_model(self):
        return self._best_model

    def save_model(self) -> tuple:
        # Generate a file name based on the current date and time
        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        vector_file_name = f"vector-{now}.pkl"
        class_file_name = f"class-{now}"

        vector_file_path = os.path.abspath(os.path.join(".", "vector", vector_file_name))
        class_file_path = os.path.abspath(os.path.join(".", "class", class_file_name))

        # Save the fitted vectorizer as a pickle file
        with open(vector_file_name, 'wb') as f:
            pickle.dump(self._count_vectorizer, f)

        # Save the best model to a file
        save_model(self._best_model, class_file_path)
        return (vector_file_path, class_file_path)


