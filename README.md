# Heart Disease Prediction

Lightweight end-to-end example that trains and serves a Random Forest model to predict heart disease from tabular patient data.

This repository includes:
- Training script (`train.py`) that performs K-Fold validation and saves a model + encoder.
- A pre-trained model file (`rf_model_40_trees_depth_10_min_samples_leaf_1.bin`).
- A Flask-based prediction endpoint (`predict.py`) and a small test client (`predict_test.py`).

## Main parts (summary)

- Dataset: `Data/heart.csv` (patient features + `HeartDisease` target).
- Model: RandomForestClassifier with saved One-Hot encoder (DictVectorizer-style) and classifier stored together in a `.bin` file.
- API: POST `/predict` accepts a single patient JSON and returns `heart_disease_probability` and `heart_disease` (bool).

## Quickstart (minimal)

1. Use Docker (recommended, ensures correct Python and deps):

```bash
docker build -t heart-predict:latest .
docker run -p 9696:9696 heart-predict:latest
```

2. Test the running server (from another shell):

```bash
python predict_test.py
```

Or use curl with a JSON body (example below).

## API — example

Endpoint: POST http://localhost:9696/predict

Sample input JSON (keys must match training features):

```json
{
	"Age": 46,
	"Sex": "M",
	"ChestPainType": "ATA",
	"RestingBP": 167,
	"Cholesterol": 163,
	"FastingBS": 0,
	"RestingECG": "ST",
	"MaxHR": 103,
	"ExerciseAngina": "N",
	"Oldpeak": 1.5,
	"ST_Slope": "Down"
}
```

Sample response:

```json
{
	"heart_disease_probability": 0.723,
	"heart_disease": true
}
```

Note: predicted fields are a float probability and a boolean decision using a 0.5 threshold.

## Installation (local, non-Docker)

The project uses `Pipfile` (Python 3.12). To install with pipenv:

```bash
pip install pipenv
pipenv install --deploy --system
```

Alternatively create a virtualenv and install dependencies derived from the Pipfile.

Start app locally:

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

Then run `python predict_test.py` to send a sample request.

## Training

Run full training and save a new model file:

```bash
python train.py
```

What happens:
- Reads `Data/heart.csv` and auto-detects categorical vs numerical columns.
- Runs K-Fold cross-validation and prints per-fold accuracy.
- Retrains on the full training set and writes the encoder + model to a `.bin` file.

## Files and structure

- `Data/heart.csv` — dataset.
- `train.py` — training + CV script.
- `predict.py` — Flask app (loads `.bin` file and serves `/predict`).
- `predict_test.py` — example client that POSTs a sample patient.
- `rf_model_40_trees_depth_10_min_samples_leaf_1.bin` — included saved model.
- `Dockerfile`, `Pipfile`, `Pipfile.lock` — runtime and packaging.

## Notes & guidance

- Ensure JSON keys and categorical values you send to `/predict` are identical (names & categories) to those used during training — otherwise the encoder may produce different feature vectors or raise an error.
- `train.py` contains a small LabelEncoder snippet that is unused in the training flow; if you modify preprocessing, make it explicit and keep it consistent between training and inference.
- There are no automated tests; consider adding a small unit test that loads the `.bin` and checks prediction output types/shape.




