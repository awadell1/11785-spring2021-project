
# Train the model using an Spot Instance
spot-train-%: src/%.py conda.install ${TRAIN_FILES}
	$(CONDA_ACTIVATE)
	./aws/spot_watcher.sh python -m src.$* $(ARGS)
	$(S3_PUSH_RUNS)
	sudo shutdown -h now

spot-train-sweep: sweep-cmd conda.install ${TRAIN_FILES}
	$(CONDA_ACTIVATE)
	./spot_watcher.sh $(SWEEP_CMD)
	$(S3_PUSH_RUNS)
	sudo shutdown -h now

.PHONY: sweep-cmd
sweep-cmd:
	[ ! -z "$(SWEEP_CMD)" ] || exit 1
