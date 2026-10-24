#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo28
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
export MLFLOW_EXPERIMENT_NAME ?= mibici-equipo28

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Run the ingest entry point with MLflow
.PHONY: mlflow-ingest
mlflow-ingest:
	mlflow run . -e ingest --env-manager local


## Run the full workflow with MLflow
.PHONY: mlflow-workflow
mlflow-workflow:
	mlflow run . -e workflow --env-manager local


## Run the P1 regression models with MLflow
.PHONY: mlflow-p1
mlflow-p1:
	mlflow run . -e p1_train --env-manager local


## Run the P2 steps (catalog, targets, models) with MLflow
.PHONY: mlflow-p2
mlflow-p2:
	mlflow run . -e p2_catalog --env-manager local
	mlflow run . -e p2_targets --env-manager local
	mlflow run . -e p2_train --env-manager local

## Run the P3 clustering with MLflow
.PHONY: mlflow-p3
mlflow-p3:
	mlflow run . -e p3_train --env-manager local


# Pendiente: descomentar junto con su entry point en MLproject.
# Se usa "#" simple para que `make help` no lo liste.

# # Run the predict entry point with MLflow
# .PHONY: mlflow-predict
# mlflow-predict:
# 	mlflow run . -e predict -P model_uri=$(MODEL_URI) --env-manager local


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
