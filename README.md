# ML Model Inference with FastAPI — Iris Classifier

A minimal FastAPI service that serves a trained scikit-learn model for **Iris flower classification**.
This implementation follows your class assignment requirements: choose a problem, train & save a model, implement FastAPI endpoints with Pydantic validation, and document + test via `/docs`.

## Problem Description
- **Task**: Predict iris species — `setosa`, `versicolor`, or `virginica` — from 4 numeric features.
- **Features**: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (in cm).
- **Dataset**: Built-in `sklearn.datasets.load_iris()`.

## Model Choice (Justification)
- **LogisticRegression + StandardScaler** inside an sklearn **Pipeline**:
  - Fast to train and predict.
  - Works well on small, linearly-separable datasets.
  - Keeps preprocessing + model in a single artifact (`model.pkl`) for easy loading in the API.

## Repository Structure
```text
fastapi-ml-iris/
├─ main.py             # FastAPI app: /, /predict, /model-info
├─ train_model.py      # Trains the model and writes model.pkl
├─ requirements.txt    # Dependencies
└─ model.pkl           # Generated after running train_model.py
```

## Setup
```bash
# 1) (Optional) create and activate a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

## Train the Model
```bash
python train_model.py
# Outputs: model.pkl and prints test accuracy
```

## Run the API
```bash
uvicorn main:app --reload
# Open: http://127.0.0.1:8000/docs
```

## Endpoints
- `GET /` — Health check.
- `POST /predict` — Predict iris species from 4 numeric features.
- `GET /model-info` — Returns model metadata (type, problem, features, classes, metrics).

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict"       -H "Content-Type: application/json"       -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
**Sample Response**
```json
{
  "prediction": "setosa",
  "confidence": 0.98
}
```

## Testing & Documentation
- Use the automatic docs: **`/docs`** (Swagger UI) and **`/redoc`**.
- Try at least two example requests (see above) with different values.
- Confirm error handling (e.g., send a string where a number is expected).

## Assumptions & Limitations
- Basic model intended for demonstration; not optimized for production.
- No database or authentication layer.
- Feature values must be realistic measurements (cm).

## How to Submit (Suggested)
1. Create a new GitHub repo and push the three core files:
   - `main.py`, `model.pkl` (after training), `requirements.txt`.
2. Include this `README.md` in the repo root.
3. Share the GitHub link as your deliverable/report.

## Grading Checklist Mapping
- **Model Implementation**: proper training/evaluation, saved as `model.pkl`.
- **FastAPI Implementation**: `/`, `/predict`, `/model-info` with Pydantic validation & error handling; model loads on startup.
- **Code Quality**: clean, commented, minimal dependencies.
- **Documentation**: this `README.md` + interactive API docs via `/docs`.
- **Testing**: demonstrate with Swagger UI and example `curl` calls.

## Notes
- Optional extensions: logging, batch prediction, confidence scores (already included when available).
- To pin versions for reproducibility, add exact versions in `requirements.txt`.# fastapi_ml_iris
