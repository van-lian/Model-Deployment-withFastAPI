from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import uvicorn
import os
from typing import Optional

class ObesityPredictor:
    def __init__(self):
        # Load the main model
        try:
            with open('best_rf_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            print("Warning: best_rf_model.pkl not found. Model will not work properly.")
            self.model = None
        
        # Load preprocessing components with error handling
        self.scalers_and_encoders = {}
        
        preprocessing_files = {
            'age_scaler': 'age_scaler.pkl',
            'onehot_encoder': 'onehot_encoder.pkl',
            'ordinal_encoder': 'ordinal_encoder.pkl',
            'weight_scaler': 'weight_scaler.pkl',
            'label_encoder': 'label_encoder.pkl',
            'random_forest_model': 'random_forest_model.pkl'
        }
        
        for name, filename in preprocessing_files.items():
            try:
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        self.scalers_and_encoders[name] = pickle.load(f)
                else:
                    print(f"Warning: {filename} not found. Will skip this preprocessing step.")
                    self.scalers_and_encoders[name] = None
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                self.scalers_and_encoders[name] = None

    def preprocess(self, input_df):
        """Preprocess the input dataframe"""
        try:
            # Scale numerical features if scalers are available
            if self.scalers_and_encoders.get('age_scaler') is not None:
                input_df['Age'] = self.scalers_and_encoders['age_scaler'].transform(input_df[['Age']])
            
            if self.scalers_and_encoders.get('weight_scaler') is not None:
                input_df['Weight'] = self.scalers_and_encoders['weight_scaler'].transform(input_df[['Weight']])
            
            # Add other preprocessing steps here as needed
            # For example, encoding categorical variables
            
            return input_df
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return input_df

    def predict(self, data):
        """Make prediction on input data"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])
            
            # Preprocess the data
            input_df = self.preprocess(input_df)
            
            # Make prediction
            pred = self.model.predict(input_df)
            
            # Convert prediction back to label if encoder is available
            if self.scalers_and_encoders.get('ord_enc_y') is not None:
                pred_label = self.scalers_and_encoders['ord_enc_y'].inverse_transform(
                    np.array(pred).reshape(-1, 1)
                )[0][0]
            else:
                pred_label = int(pred[0])
            
            return pred_label
        except Exception as e:
            raise ValueError(f"Error making prediction: {e}")

class ObesityInput(BaseModel):
    Age: float
    Weight: float
    Height: float
    FCVC: float
    NCP: float
    FAF: float
    TUE: float
    family_history_with_overweight: str
    FAVC: str
    SMOKE: str
    SCC: str
    MTRANS: str
    Gender: str
    CALC: str
    CAEC: str

# Initialize FastAPI app
app = FastAPI(
    title="Obesity Prediction API",
    description="API for predicting obesity levels based on lifestyle and physical attributes",
    version="1.0.0"
)

# Initialize predictor
try:
    predictor = ObesityPredictor()
    print("Predictor initialized successfully")
except Exception as e:
    print(f"Error initializing predictor: {e}")
    predictor = None

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Obesity Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.model is not None
    }

@app.post("/predict")
def predict_obesity(data: ObesityInput):
    """Predict obesity level based on input features"""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check if model files are present."
        )
    
    try:
        pred_label = predictor.predict(data.dict())
        return {
            "prediction": pred_label,
            "input_data": data.dict(),
            "status": "success"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run the application
if __name__ == "__main__":
    import os
    
    # Check if model files exist
    required_files = [
        'age_scaler.pkl',
        'onehot_encoder.pkl',
        'ordinal_encoder.pkl',
        'weight_scaler.pkl',              
        'label_encoder.pkl',     
        'random_forest_model.pkl'       
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Missing model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run model.py first to generate the model files.")
        exit(1)
    
    print("Starting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",  # Match this to your actual filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )