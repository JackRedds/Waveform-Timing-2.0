NAME := wave-timing
POETRY_LOCK := poetry.lock

install: pyproject.toml $(POETRYF_LOCK)
	@echo "Creating virtual enviroment from enviroment.yml ..."
	@conda env create -f enviroment.yml
	@echo "Installing packages from $(POETRY_LOCK) ..."
	@conda run -n wave-timing poetry install
	@echo "Done installation"
