
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import sys
import glob
import pickle
import argparse
import requests
import logging
from datetime import datetime
#
# ML-related modules
#
import pandas as pd
from pycaret.classification import *
from sklearn.feature_extraction.text import CountVectorizer
#
# App-related modules
#
from flask import Flask, request, render_template, jsonify
#
# Internal modules
#
from model import MaliciousWebpageDataset
from webparse import WebpageParser
#
OPT_VERBOSE_HELP = "Display additional information about execution."
#
logger = logging.getLogger(__name__)
app = Flask(__name__)

def get_newest_file(path):
    """ Selects the most recent file created from the `models` directory. """
    files = glob.glob(os.path.join(path, '*.pkl'))
    if files is None or len(files) <= 0:
        return None
    return max(files, key=os.path.getctime)

# Check if were are currently running on Heroku
if 'DYNO' in os.environ:
    # if so load the model on execution
    vector_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "vector"))
    class_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "class"))
    logger.debug(f"Latest model (Vector): {vector_model_file}")
    logger.debug(f"Latest model (Classification): {class_model_file}")
    if vector_model_file is not None:
        with open(f"{vector_model_file}", 'rb') as f:
            current_vector_model = pickle.load(f)
    else:
        raise Exception(f"No vector model found.")
    
    if class_model_file is not None:
        current_class_model = load_model(class_model_file.split('.', maxsplit=1)[0])
    else:
        raise Exception(f"No classification model found.")
else: # Otherwise let the user decide how to load the model.
    current_vector_model = None
    current_class_model = None
    

@app.route('/')
def home():
    """ Shows the home page using the 'templates/home.html' page """
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
    """ Endpoint processing the data to predict the quality. """
    # Specifies whether the operation succeed
    success = False
    # Error message of label of the website
    label = "N/A"
    # Get the URL requested
    url = request.form["url"]
    # Define a semi-legitimate user agent string
    ua = "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_5; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.204"
    # Ensure the URL is valid
    # TODO: use validators module
    if url is not None and len(url) > 0:
        try:
            # Try to request the URL requests
            response = requests.get(url, headers={"User-Agent": ua}, verify=False, timeout=10)
            # Check if we got contents
            if response.status_code == 200:
                html = response.text
                logger.info(f"[+] Received {len(html)} byte(s) from '{url}'.")

                wp = WebpageParser()
                f = wp.parse_html(html)

                if f["is_english"] is True:
                    global current_vector_model
                    global current_class_model

                    # Generate a `DataFrame` from the features extracted
                    df_features = pd.DataFrame([f])

                    # Drop unneeded columns
                    df_features.drop('is_english', axis=1, inplace=True)
                    df_features.drop('title_raw', axis=1, inplace=True)

                    # Vectorize the website content using the pre-trained count vectorizer
                    X = current_vector_model.transform(df_features["text_clean"])

                    df_words = pd.DataFrame(data=X.toarray(), columns = current_vector_model.get_feature_names())

                    # Merge dataframes together
                    df = pd.concat([df_features, df_words], axis=1)

                    prediction = predict_model(current_class_model, data=df)
                    label = prediction.iloc[-1]["Label"]
                    success = True
                    
                    logger.info(f"[+] Classification: {label.upper()}")
                else:
                    logger.error(f"[-] Target website is not written in English.")
            else:
                logger.error(f"[-] Received response '{response.status_code}' from '{url}'.")
        except Exception as e:
            label = f"Unable to reach '{url}': {str(e)}"
            logger.error(label)

    if not success:
        logger.error(label)

    return jsonify({
        "ok": success,
        "label": label
    })

def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="")
    parser.add_argument('-t', '--train',
                    dest="do_train",
                    action="store_true",
                    help="Train a new model without starting the web server.")
    parser.add_argument('--parse',
                    dest="data_dir",
                    help="Parses HTML webpages in the given directory.")
    parser.add_argument('--class',
                    dest="data_class",
                    help="Classification of the data contained in the the directory specified in the `--parse` option.")
    parser.add_argument('-p', '--port',
                    dest="port",
                    default=5000,
                    help="Port to listen on for HTTP requests.")
    parser.add_argument('-m', '--model',
                    dest="model",
                    required=False,
                    help="Model file to use.")
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action="store_true",
                        dest="is_debug",
                        help=OPT_VERBOSE_HELP)

    logging.basicConfig(level=logging.DEBUG)

    # Parse command-line arguments
    args = parser.parse_args(args=argv[1:])
    print("")

    if args.data_dir is not None:
        if os.path.isdir(args.data_dir):
            logger.info(f"Parsing webpages in '{args.data_dir}'.")
            wp = WebpageParser(_path=args.data_dir)
            wp.parse(_tag=args.data_class)
        else:
            logger.error(f"Directory not found: {args.data_dir}")
    # Train a new model
    elif args.do_train:
        # Load the data
        datafile = os.path.abspath('./data/sample/data.csv')
        dataset = MaliciousWebpageDataset(datafile)
        #logger.info(f"{dataset.nb_rows} row(s) loaded from '{datafile}'.")
        (vf, md) = dataset.save_model()
        logger.info(f"Saved vector model to '{vf}.")
        logger.info(f"Saved classification model to '{md}.")
    # Otherwise, start the server
    else:
        # Load the latest model
        vector_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "vector"))
        class_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "class"))
        if vector_model_file is not None and class_model_file is not None:
            logger.info(f"Selected '{vector_model_file}' as vector model.")
            logger.info(f"Selected '{class_model_file}' as classification model.")
            # For some reason, `load_model` appends `.pkl` to the file, so 
            # we need to remove it.
            global current_vector_model
            global current_class_model

            with open(f"{vector_model_file}", 'rb') as f:
                current_vector_model = pickle.load(f)

            current_class_model = load_model(class_model_file.split('.', maxsplit=1)[0])
            # Start the server
        else:
            logger.warning(f"No model available.")
        app.run(port=int(args.port), debug=bool(args.is_debug), threaded=True)

def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))

if __name__ == '__main__':
    entry_point()

