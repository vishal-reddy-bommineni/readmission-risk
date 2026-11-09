.PHONY: setup features train all app smoke clean

setup:
	pip install -r requirements.txt

features:
	python -m src.build_features

train:
	python -m src.train --n-boot 1000

all: features train

app:
	streamlit run src/app.py

# Runs the whole pipeline on FAKE data to check the code works. Never report these numbers.
smoke:
	python scripts/make_smoke_test_data.py data/raw/smoke_test.csv
	RAW_OVERRIDE=data/raw/smoke_test.csv python -m src.build_features
	python -m src.train --n-boot 50 --skip-leakage

clean:
	rm -rf data/processed/* models/* reports/*.csv reports/*.json reports/RESULTS.md reports/figures/* dashboard/*.csv
