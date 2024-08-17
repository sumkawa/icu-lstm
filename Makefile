.PHONY: install create-env update test test-ruff test-pycodestyle

ENV_NAME=icu_classifier_env
PYTHON_VERSION=3.11

install: create-env
	bash -c "source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install -r requirements.txt"

create-env:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y

update:
	bash -c "source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pip install --upgrade pip && pip install --upgrade -r requirements.txt"

test: test-pycodestyle test-ruff

test-ruff:
	bash -c "source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && ruff src --fix"

test-pycodestyle:
	bash -c "source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && pycodestyle --ignore=E501,W191,W293,E302,E305,W291,W292,E126,E265,E226,E262,E261,W391,E121,E123,E712,E23,W504,E402,W605 src"
