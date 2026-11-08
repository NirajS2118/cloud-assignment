.PHONY: up down train test lint clean

up:
	@echo "Creating k3d cluster..."
	k3d cluster create --config cluster/k3d.yaml || true
	@echo "Deploying SeaweedFS storage..."
	kubectl apply -f k8s/storage/
	@echo "Deploying command-parser service in app namespace..."
	kubectl apply -f k8s/app/

down:
	@echo "Deleting k3d cluster..."
	k3d cluster delete cloud-assignment

train:
	@echo "Running model training..."
	python training/train.py $(if $(SUBSET),--subset $(SUBSET),)

test:
	@echo "Running unit tests..."
	python -m unittest discover -s tests -p "test_*.py" -v

lint:
	@echo "Running linter..."
	python -m flake8 service/ worker/ training/ tests/ --max-line-length=120 --ignore=E501,W503 || python -m ruff check service worker training tests || true

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

