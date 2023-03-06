#!/usr/bin/env bash

python app.py --parse "${1}/malicious" --class malicious
python app.py --parse "${1}/benign" --class benign
cp ./data/data-benign.csv ./data/data.csv
tail -n +2 ./data/data-malicious.csv >> ./data/data.csv
wc -l ./data/data.csv
