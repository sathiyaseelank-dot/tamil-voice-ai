.PHONY: setup demo train clean

PYTHON ?= python3
VENV ?= .venv

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(VENV)/bin/pip install -r requirements.txt

demo:
	$(VENV)/bin/python demo.py

train:
	$(VENV)/bin/python train.py --data_dir tokenized_data --epochs 5 --batch_size 4

clean:
	rm -rf __pycache__ */__pycache__ *.egg-info build dist
