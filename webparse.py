#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import re
import nltk
import string
import logging
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)
#
# Define a function to check the method attribute
def is_post(form):
    return form.get('method', '').lower() == 'post'
#
class WebpageParser(object):
    def __init__(self, _path:str) -> None:
        self._path = _path
        nltk.download('punkt')
        nltk.download('stopwords')

    @property
    def data_path(self):
        return self._path

    def parse(self, _tag):
        cols = ["title_raw", 
                "has_form", 
                "has_login_form", 
                "has_js", 
                "js_include_b64",
                "nb_tokens",
                "text_clean",
                "classification"]
        df = pd.DataFrame(columns=cols)
        # Iterate through each file in the folder and its subfolders
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                # Features of the webpage read
                features = {}
                # Check if the file is an HTML file
                if file.endswith('.html') or file.endswith('.htm'):
                    logger.debug(f"\tProcessing '{file}'.")
                    # Read the HTML file and parse it using BeautifulSoup
                    with open(os.path.join(root, file), 'r') as f:
                        html = f.read()
                        soup = BeautifulSoup(html, 'html.parser')
                        # get the title of the webpage
                        if soup.title is not None:
                            wp_title = soup.title.string.strip().lower()
                            features["title_raw"] = " ".join(nltk.word_tokenize(wp_title))
                        else:
                            features["title_raw"] = ""
                        # Extract the text content from the HTML
                        text = soup.get_text()

                        # search for a login form
                        forms = soup.find_all('form', {'method': 'POST'})
                        forms += soup.find_all('form', {'method': 'post'})
                        features["has_form"] = len(forms) > 0
                        features["has_login_form"] = len(soup.find_all("input", {"type": "password"})) > 0
                        #for form in forms:
                        #    features["has_login_form"] |= (len(form.find("input", {"type": "password"})) > 0)

                        # Check if JavaScript with base64 strings exists
                        scripts = soup.find_all('script')
                        features["has_js"] = (len(scripts) > 0)
                        features["js_include_b64"] = False
                        for script in scripts:
                            if script.has_attr('src'):
                                # Check if script is loaded from external source
                                script_content = None
                            else:
                                script_content = script.string

                            if script_content is not None and re.search("data:[a-zA-Z0-9+/]+={0,2}", script_content):
                                features["js_include_b64"] |= True
                    
                        # tokenize the text after convertign to lowercase
                        tokens = nltk.word_tokenize(text.lower().strip())

                        # remove stop words and punctuation
                        stop_words = set(stopwords.words('english'))
                        clean_tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
                        # Save the number of tokens
                        features["nb_tokens"] = len(clean_tokens)

                        # join the cleaned tokens back into a string
                        features["text_clean"] = ' '.join(clean_tokens)
                        # add the classification to the row
                        features["classification"] = _tag
                        # add a new row to the DataFrame
                        df = df.append(features, ignore_index=True)
        
        # save to CSV
        df.to_csv(f"./data/data-{_tag}.csv", index=False)