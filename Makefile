install:
	pip install -r requirements.txt

train:
	sh run.wh

inference:
	python inference.py

format:
	#black *.py

lint:
	#pylint --disable=R,C *.py


all: install train inference