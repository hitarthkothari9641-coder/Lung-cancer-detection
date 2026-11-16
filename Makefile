.PHONY: install preprocess train eval backend frontend demo test docker-up docker-down

install:
	pip install -r backend/requirements.txt -r training/requirements.txt
	cd frontend && npm install

preprocess:
	python training/preprocess.py

train:
	python training/train.py

eval:
	python training/evaluate.py

backend:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

demo:
	@echo "Starting Lung Nodule Grader Demo stack..."
	python scripts/make_sample_data.py
	python training/evaluate.py
	@echo "Backend starting on http://localhost:8000"
	@echo "Frontend starting on http://localhost:3000"
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

test:
	python -m pytest backend/tests -v

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
