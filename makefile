
.ONESHELL:
SHELL=/bin/bash

# Macro to activate conda environment
CONDA_ENV = idl-project
define CONDA_ACTIVATE
	@source "$(shell conda info --base)/etc/profile.d/conda.sh"
	conda activate $(CONDA_ENV)
endef

conda.install: environment.yml
	conda env update -n $(CONDA_ENV) -f $< | tee $@

# Fetch files from s3
data/%:
	aws s3 sync "s3://11785-spring2021-project/data/$*" $@

# Only use a subset of the training data for testing
TEST_FILES := data/Brats17TrainingData/HGG/Brats17_2013_2_1
TEST_FILES += data/Brats17TrainingData/LGG/Brats17_2013_0_1

# Use the full set for training
TRAIN_FILES := data/Brats17TrainingData

# Run Tests
PHONY: test test-coverage
test: conda.install $(TEST_FILES)
	$(CONDA_ACTIVATE)
	python3 -m pytest tests

test-coverage: conda.install $(TEST_FILES)
	$(CONDA_ACTIVATE)
	python3 -m pytest --cov-report xml tests

train-% : src/%.py conda.install
	$(CONDA_ACTIVATE)
	python -m src.$* $(ARGS)

include aws/makefile
