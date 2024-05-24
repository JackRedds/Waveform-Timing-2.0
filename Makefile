NAME := wave-timing
CONDA_ENV := enviroment.yml
POETRY_LOCK := poetry.lock


install: pyproject.toml $(POETRY_LOCK)
	@echo "Creating virtual enviroment from $(CONDA_ENV) ..."
	@conda env create -f enviroment.yml
	@echo "Installing packages from $(POETRY_LOCK) ..."
	@conda run -n wave-timing poetry install
	@echo "Done installation"
