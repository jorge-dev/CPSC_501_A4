SHELL := /bin/bash

VENV := .venv

all:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate
	$(VENV)/bin/pip install numpy idx2numpy matplotlib

.PHONY: all mnist mnist-load no-mnist no-mnist-load heart heart-load clean
mnist: 
	$(VENV)/bin/python3 MNIST_starter.py

mnist-load:
	./$(VENV)/bin/python3 MNIST_starter.py --load neural-nets/part1.pkl

no-mnist:
	./$(VENV)/bin/python3 notMNIST_starter.py

no-mnist-load:
	./$(VENV)/bin/python3 notMNIST_starter.py --load neural-nets/part2.pkl
	
heart:
	./$(VENV)/bin/python3 heart_starter.py

heart-load:
	./$(VENV)/bin/python3 heart_starter.py --load neural-nets/part3.pkl

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete



