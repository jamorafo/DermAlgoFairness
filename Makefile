IMAGE_NAME=dermalgo-fairness:arxiv-revision-tf2.13-cuda11.8
CONTAINER_NAME=dermalgo-arxiv-revision
PROJECT_DIR=/workspace/DermAlgoFairness

.PHONY: build shell record-env dry-run-resnet50

build:
	docker build -t $(IMAGE_NAME) .

shell:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		--gpus all \
		--memory=10g \
		--shm-size=4g \
		-v $(HOME)/DermAlgoFairness:$(PROJECT_DIR) \
		$(IMAGE_NAME) \
		bash

record-env:
	docker run --rm -it \
		--gpus all \
		--memory=10g \
		--shm-size=4g \
		-v $(HOME)/DermAlgoFairness:$(PROJECT_DIR) \
		$(IMAGE_NAME) \
		bash -lc "cd $(PROJECT_DIR) && PYTHONPATH=src python3 scripts/00_record_environment.py"

dry-run-resnet50:
	docker run --rm -it \
		--gpus all \
		--memory=10g \
		--shm-size=4g \
		-v $(HOME)/DermAlgoFairness:$(PROJECT_DIR) \
		$(IMAGE_NAME) \
		bash -lc "cd $(PROJECT_DIR) && PYTHONPATH=src python3 scripts/03_train_model.py --config configs/resnet50.yaml --seed 347535239 --dry-run"
