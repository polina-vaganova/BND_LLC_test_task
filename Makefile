# Name of a virtual environment:
VENV_NAME=.venv


# Detect OS and set paths:
ifeq ($(OS), Windows_NT)
    # Windows:
    PIP=$(VENV_NAME)/Scripts/pip
    PYTHON=$(VENV_NAME)/Scripts/python
    CLEAN_CMD=if exist $(VENV_NAME) rmdir /s /q $(VENV_NAME)
else
    # MacOS & Ubuntu:
    PIP=$(VENV_NAME)/bin/pip
    PYTHON=$(VENV_NAME)/bin/python
    CLEAN_CMD=rm -rf $(VENV_NAME)
endif

check:
	python3 --version || python --version
	@echo "Python locations:"
	@command -v python3 || command -v python || where python3 || where python || echo "Python not found"

venv:
	python3 -m venv $(VENV_NAME) || python -m venv $(VENV_NAME)

install: venv
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

clear:
	$(CLEAN_CMD)
