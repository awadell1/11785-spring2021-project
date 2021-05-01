
# Train the model using an Spot Instance
spot-train-%: aws-train-%
	sudo shutdown -h now

aws-train-%: src/%.py conda.install ${TRAIN_FILES} mkswap
	$(CONDA_ACTIVATE)
	python -m src.$* $(ARGS)

spot-train-sweep: sweep-cmd conda.install ${TRAIN_FILES} mkswap
	$(CONDA_ACTIVATE)
	$(SWEEP_CMD)

.PHONY: sweep-cmd
sweep-cmd:
	[ ! -z "$(SWEEP_CMD)" ] || exit 1

.PHONY:

# Increase swap space to ~40 GB
mkswap: SWAP_COUNT ?= 312
mkswap:
	set -e
	sudo dd if=/dev/zero of=/swapfile bs=128M count=$(SWAP_COUNT) status=progress
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
