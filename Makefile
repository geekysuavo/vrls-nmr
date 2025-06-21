.PHONY: clean again test

all: build

build:
	python setup.py build
	python setup.py bdist_wheel
	pip install dist/*.whl

clean:
	pip uninstall vrlsnmr -y
	rm -rvf build dist vrlsnmr.egg-info .ipynb_checkpoints
	rm -rvf vrlsnmr/__pycache__ test/__pycache__
	rm -vf vrlsnmr/_ops.so

again: clean build
