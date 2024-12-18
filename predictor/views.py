from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd

def home(request):
    prediction = None
    
    if request.method == 'POST':
        # Load the model and scaler
        model = joblib.load('predictor/ml_model/rf_model.joblib')
        scaler = joblib.load('predictor/ml_model/scaler.joblib')
        
        # Get form data
        size = float(request.POST['size'])
        bhk = float(request.POST['bhk'])
        bathroom = float(request.POST['bathroom'])
        area_type = request.POST['area_type']
        furnishing = request.POST['furnishing']
        city = request.POST['city']
        
        # Initial prediction for price_per_sqft calculation
        # Using median values from training data based on city and property type
        if city == 'Mumbai':
            base_rent = 65000
        elif city == 'Delhi':
            base_rent = 45000
        elif city == 'Bangalore':
            base_rent = 35000
        elif city == 'Chennai':
            base_rent = 30000
        elif city == 'Hyderabad':
            base_rent = 28000
        else:  # Kolkata
            base_rent = 25000
            
        price_per_sqft = base_rent / size
        room_bath_ratio = bhk / bathroom
        
        # Market sophistication metrics
        luxury_score = size * price_per_sqft * bathroom
        efficiency_ratio = size / (bhk + bathroom)
        market_power = luxury_score * efficiency_ratio
        
        # Interaction terms
        size_bath = size * bathroom
        size_bhk = size * bhk
        premium_location_score = price_per_sqft * market_power
        
        # Create one-hot encoded features
        area_type_features = [
            1 if area_type == "Built Area" else 0,
            1 if area_type == "Carpet Area" else 0,
            1 if area_type == "Super Area" else 0
        ]
        
        furnishing_features = [
            1 if furnishing == "Furnished" else 0,
            1 if furnishing == "Semi-Furnished" else 0,
            1 if furnishing == "Unfurnished" else 0
        ]
        
        city_features = [
            1 if city == "Kolkata" else 0,
            1 if city == "Mumbai" else 0,
            1 if city == "Bangalore" else 0,
            1 if city == "Delhi" else 0,
            1 if city == "Chennai" else 0,
            1 if city == "Hyderabad" else 0
        ]
        
        # Combine all features
        features = np.array([[
            size, bhk, bathroom,
            price_per_sqft, room_bath_ratio,
            luxury_score, efficiency_ratio, market_power,
            size_bath, size_bhk, premium_location_score,
            *area_type_features,
            *furnishing_features,
            *city_features
        ]])
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
    return render(request, 'predictor/home.html', {'prediction': prediction})
