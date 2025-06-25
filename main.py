from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import Optional

MODEL_DIR = "models"

class ObesityPredictor:
    def __init__(self):
        # Load all models and encoders
        self.components = {}
        for name in ['best_rf_model', 'age_scaler', 'weight_scaler', 'onehot_encoder', 
                    'ordinal_encoder', 'label_encoder', 'expected_features']:
            with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'rb') as f:
                self.components[name] = pickle.load(f)

    def preprocess(self, data):
        try:
            df = pd.DataFrame([data])
            
            # Scale numerical features
            df['Age'] = self.components['age_scaler'].transform(df[['Age']])
            df['Weight'] = self.components['weight_scaler'].transform(df[['Weight']])
            
            # One-hot encode categorical features
            onehot_cols = ['MTRANS', 'Gender']
            if all(col in df.columns for col in onehot_cols):
                onehot_df = pd.DataFrame(
                    self.components['onehot_encoder'].transform(df[onehot_cols]), 
                    columns=self.components['onehot_encoder'].get_feature_names_out(onehot_cols)
                )
                
                df = df.drop(columns=onehot_cols)
                df = pd.concat([df.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)
            
            # Ordinal encode CALC and CAEC using the combined encoder
            ordinal_cols = ['CALC', 'CAEC']
            present_ordinal_cols = [col for col in ordinal_cols if col in df.columns]
            
            if len(present_ordinal_cols) == 2:
                try:
                    ordinal_transformed = self.components['ordinal_encoder'].transform(df[ordinal_cols])
                    df[ordinal_cols] = ordinal_transformed
                    
                except Exception as e:
                    # Manual fallback encoding based on your training data
                    calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
                    caec_mapping = {'Sometimes': 0, 'Frequently': 1, 'no': 2, 'Always': 3}
                    
                    df['CALC'] = df['CALC'].map(calc_mapping).fillna(0).astype(int)
                    df['CAEC'] = df['CAEC'].map(caec_mapping).fillna(0).astype(int)
                    
            else:
                # Set default values for missing columns
                for col in ordinal_cols:
                    if col not in df.columns:
                        df[col] = 0
            
            # Binary encode yes/no features
            binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
            for col in binary_cols:
                if col in df.columns:
                    df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
            
            # Align with expected features
            expected_features = self.components['expected_features']
            
            # Add missing features with default values
            for feat in expected_features:
                if feat not in df.columns:
                    df[feat] = 0
            
            # Remove extra columns and reorder to match expected features
            df = df[expected_features]
            
            return df
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    def predict(self, data):
        X = self.preprocess(data)
        pred = self.components['best_rf_model'].predict(X)
        
        # The label_encoder is actually an OrdinalEncoder, so it needs 2D input
        try:
            if pred.ndim == 1:
                pred_2d = pred.reshape(-1, 1)  # Convert 1D to 2D
            else:
                pred_2d = pred
            
            result = self.components['label_encoder'].inverse_transform(pred_2d)[0][0]
            return result
        except Exception as e:
            # Fallback: return the raw prediction
            return pred[0]

class ObesityInput(BaseModel):
    Age: float
    Weight: float
    Height: float
    FCVC: float
    NCP: float
    CH2O: Optional[float] = 2.0
    FAF: float
    TUE: float
    family_history_with_overweight: str  # yes/no
    FAVC: str  # yes/no
    SMOKE: str  # yes/no
    SCC: str  # yes/no
    MTRANS: str  # transportation method
    Gender: str  # Male/Female
    CALC: str  # alcohol consumption frequency
    CAEC: str  # eating between meals

# Initialize FastAPI app
app = FastAPI(
    title="Obesity Prediction API", 
    description="API for predicting obesity levels based on lifestyle factors",
    version="1.0.0"
)

# Add CORS middleware for Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = ObesityPredictor()
    model_loaded = True
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    predictor = None
    model_loaded = False

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Obesity Prediction API", 
        "status": "running",
        "endpoints": ["/predict", "/health"]
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy" if model_loaded else "error",
        "model_loaded": model_loaded
    }

@app.post("/predict", tags=["Prediction"])
def predict_obesity(data: ObesityInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction = predictor.predict(data.dict())
        return {
            "prediction": prediction,
            "confidence": "Model trained on lifestyle and demographic factors",
            "input_data": data.dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info", tags=["Info"])
def model_info():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest",
        "features_required": [
            "Age", "Weight", "Height", "FCVC", "NCP", "CH2O", "FAF", "TUE",
            "family_history_with_overweight", "FAVC", "SMOKE", "SCC", 
            "MTRANS", "Gender", "CALC", "CAEC"
        ],
        "categorical_values": {
            "family_history_with_overweight": ["yes", "no"],
            "FAVC": ["yes", "no"],
            "SMOKE": ["yes", "no"], 
            "SCC": ["yes", "no"],
            "Gender": ["Male", "Female"],
            "MTRANS": ["Public_Transportation", "Walking", "Automobile", "Bike"],
            "CALC": ["no", "Sometimes", "Frequently", "Always"],
            "CAEC": ["no", "Sometimes", "Frequently", "Always"]
        }
    }

# For Azure Web App
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
