# California House Price Prediction API

A REST API that predicts California housing prices using a Random Forest model trained on the classic California Housing dataset. Supports both single-record predictions and bulk predictions via CSV upload.

## What it does

- Trained on **20,640 records** with **8 features** (median income, house age, average rooms/bedrooms, population, occupancy, latitude, longitude).
- Achieves an **average prediction error (MAE) of ~$39,000**, giving every prediction a realistic confidence range.
- Exposes both a **single-prediction endpoint** and a **batch CSV upload endpoint** that returns a downloadable CSV of results.

## How it works

- **Model:** `RandomForestRegressor` (100 trees) from scikit-learn, trained on an 80/20 train-test split.
- **API:** Built with **FastAPI**, using Pydantic models to validate input ranges (e.g., valid latitude/longitude bounds, positive values only).
- **Persistence:** The trained model and feature list are saved with `joblib` and loaded at startup for fast inference.

## Project structure

```
house_prediction_api/
├── train.py                      # Trains the model and saves it
├── explore.py                    # Dataset exploration/EDA
├── main.py                       # FastAPI app with prediction endpoints
├── house_model.joblib            # Trained RandomForestRegressor (tracked via Git LFS)
├── house_features.joblib         # Saved feature column order
├── house_prediction_inputs.csv  # Sample input file for batch predictions
└── Requirements.txt
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/house-prediction-api.git
cd house-prediction-api

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model from scratch
python train.py

# 5. Run the API
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are auto-generated at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method | Endpoint         | Description                                           |
|--------|------------------|--------------------------------------------------------|
| GET    | `/`              | Health/welcome message                                 |
| GET    | `/health`        | Model status, feature list, and average error          |
| POST   | `/predict`       | Predict price for a single house (JSON body)            |
| POST   | `/predict_file`  | Upload a CSV of houses, get back a CSV with predictions |

### Example: `/predict` request body

```json
{
  "MedInc": 8.3,
  "HouseAge": 41,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## Tech stack

- Python
- FastAPI
- scikit-learn (RandomForestRegressor)
- Pandas
- Pydantic
- joblib
