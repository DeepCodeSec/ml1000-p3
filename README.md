# ml1000-p2
Project 2 for the ML1000 course

## Installation

To install the _Python_ program locally, first create a virtual environment:

```sh
$ python3 -m venv venv
$ source ./bin/venv/activate
$ (venv)
```

Next, install the required modules:

```sh
$ (venv) pip install -r requirements.txt
```

## Usage

```sh
$python3 app.py -h
usage: app.py [-h] [-t] [-p PORT] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Train a new model without starting the web server.
  -p PORT, --port PORT  Port to listen on for HTTP requests.
  -v, --verbose         Display additional information about execution.
```

To run the server locally, simply use the following command:

```sh
$ (venv) python3 app.py
```


## Usage

## Application

* [https://app4appts.herokuapp.com/](https://app4appts.herokuapp.com/)

## References
