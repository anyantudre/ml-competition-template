install:
	pip install -r requirements.txt

folds:
	python ./src/create_folds.py
	
train:
	PYTHONPATH=./src sh ./src/run.sh

inference:
	python ./src/inference.py

format:
	#black *.py

lint:
	#pylint --disable=R,C *.py


all: install train inference