
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
from pycaret.classification import *
from pycaret.nlp import assign_model
import pandas as pd
from flask import Flask, request, render_template, jsonify
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
    text_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "text"))
    class_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "class"))
    logger.debug(f"Latest model (Text Processing): {text_model_file}")
    logger.debug(f"Latest model (Classification): {class_model_file}")
    if text_model_file is not None:
        current_text_model = load_model(text_model_file.split('.', maxsplit=1)[0])
    else:
        raise Exception(f"No text processing model found.")
    
    if class_model_file is not None:
        current_class_model = load_model(class_model_file.split('.', maxsplit=1)[0])
    else:
        raise Exception(f"No text processing model found.")
else: # Otherwise let the user decide how to load the model.
    current_text_model = None
    current_class_model = None

@app.route('/')
def home():
    """ Shows the home page using the 'templates/home.html' page """
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
    """ Endpoint processing the data to predict the quality. """
    success = False
    label = "N/A"
    # Extract the data from the form
    logger.debug(request.form)
    url = request.form["url"]
    if url is not None and len(url) > 0:
        try:
            # Try to request the URL requests
            response = requests.get(url, headers={}, verify=False)
            # If we were able to retrieve the contents of the URL, proceed
            if response.status_code == 200:
                # get the html of the page
                html = response.text
                # parse the HTML and extract the features
                parser = WebpageParser()
                try:
                    f = parser.parse_html(html)
                    d = pd.DataFrame([f])
                    if d["is_english"] == True:
                        # Load the features in the model.
                        global current_text_model
                        global current_class_model
                        if current_text_model is not None:
                            exp_name = setup(data=d, target='text_clean')
                            # Apply the LDA model to the unseen document to extract topics
                            topics = assign_model(current_text_model)
                            topics.drop(['title_raw', 'text_clean', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace = True)
                            prediction = predict_model(current_class_model, data=topics)
                            label = prediction.iloc[-1]["Label"]
                            success = True
                        else:
                            logger.error(f"No model defined.")
                    else:
                        label = "The language of the webpage provided is not supported."
                except Exception as e:
                    label = f"Error processing '{url}': {str(e)}"
                    logger.error(label)
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
        logger.info(f"{dataset.nb_rows} row(s) loaded from '{datafile}'.")

        # Generate a file name based on the current date and time
        now = datetime.now().strftime("%Y%m%d-%H%M%S")

        topics_file_name = f"text-{now}"
        file_path = os.path.abspath(os.path.join(".", "models", "text", topics_file_name))
        dataset.save_topics_model_to(file_path)
        logger.info(f"Save topics model to '{file_path}.")

        class_file_name = f"class-{now}"
        file_path = os.path.abspath(os.path.join(".", "models", "class", class_file_name))
        dataset.save_classifier_model_to(file_path)
        logger.info(f"Save classifier model to '{file_path}.")

    # Otherwise, start the server
    else:
        # Load the latest model
        text_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "text"))
        class_model_file = get_newest_file(os.path.join(os.getcwd(), "models", "class"))
        if text_model_file is not None and class_model_file is not None:
            logger.info(f"Selected '{text_model_file}' as text processing model.")
            logger.info(f"Selected '{class_model_file}' as classification model.")
            # For some reason, `load_model` appends `.pkl` to the file, so 
            # we need to remove it.
            global current_text_model
            global current_class_model
            current_text_model = load_model(text_model_file.split('.', maxsplit=1)[0])
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

