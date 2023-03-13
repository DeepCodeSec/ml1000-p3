install:
	sudo apt-get install python3.9 python3-dev python3-pip python3-numpy build-essential

./data/data.csv:
	python app.py --parse ./data/sample/malicious --class malicious
	python app.py --parse ./data/sample/benign --class benign
	cp ./data/data-benign.csv ./data/data.csv
	tail -n +2 ./data/data-malicious.csv >> ./data/data.csv
	wc -l ./data/data.csv

data: clean ./data/data.csv

sample:
	head -20000 ./data/full/data.csv > ./data/sample/data.csv
	tail -20000 ./data/full/data.csv >> ./data/sample/data.csv
	grep "benign" ./data/sample/data.csv | wc -l
	grep "malicious" ./data/sample/data.csv | wc -l

clean:
	-rm ./data/data-benign.csv
	-rm ./data/data-malicious.csv
	-rm ./data/data.csv
