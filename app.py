"""
Real Estate Investment Advisor - Flask Application
Predicting Property Profitability & Future Value

This application provides:
1. Classification: Is this property a "Good Investment"?
2. Regression: What will be the estimated price after 5 years?
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
import os

app = Flask(__name__)

# ============================================================
# LOAD MODELS AND ARTIFACTS
# ============================================================

MODELS_DIR = 'models'

def load_artifacts():
    """Load all saved models and artifacts"""
    artifacts = {}
    
    try:
        # Load Classification Model
        artifacts['clf_model'] = joblib.load(f'{MODELS_DIR}/classification_model.joblib')
        print("✓ Classification model loaded")
        
        # Load Regression Model
        artifacts['reg_model'] = joblib.load(f'{MODELS_DIR}/regression_model.joblib')
        print("✓ Regression model loaded")
        
        # Load Scaler
        artifacts['scaler'] = joblib.load(f'{MODELS_DIR}/scaler.joblib')
        print("✓ Scaler loaded")
        
        # Load Label Encoders
        with open(f'{MODELS_DIR}/label_encoders.pkl', 'rb') as f:
            artifacts['label_encoders'] = pickle.load(f)
        print("✓ Label encoders loaded")
        
        # Load Feature Names (SEPARATE for Classification and Regression)
        with open(f'{MODELS_DIR}/clf_feature_names.pkl', 'rb') as f:
            artifacts['clf_feature_names'] = pickle.load(f)
        print(f"✓ Classification feature names loaded ({len(artifacts['clf_feature_names'])} features, NO Price)")
        
        with open(f'{MODELS_DIR}/reg_feature_names.pkl', 'rb') as f:
            artifacts['reg_feature_names'] = pickle.load(f)
        print(f"✓ Regression feature names loaded ({len(artifacts['reg_feature_names'])} features, WITH Price)")
        
        # Keep backward compatible feature_names
        artifacts['feature_names'] = artifacts['clf_feature_names']
        
        # Load Dropdown Values
        with open(f'{MODELS_DIR}/dropdown_values.pkl', 'rb') as f:
            artifacts['dropdown_values'] = pickle.load(f)
        print("✓ Dropdown values loaded")
        
        # Load City Growth Rates
        with open(f'{MODELS_DIR}/city_growth_rates.pkl', 'rb') as f:
            artifacts['city_growth_rates'] = pickle.load(f)
        print("✓ City growth rates loaded")
        
        # Load City-State mapping
        with open(f'{MODELS_DIR}/city_state_map.pkl', 'rb') as f:
            artifacts['city_state_map'] = pickle.load(f)
        print("✓ City-State mapping loaded")
        
        # Load sample data for insights
        artifacts['sample_data'] = pd.read_csv(f'{MODELS_DIR}/sample_data.csv')
        print("✓ Sample data loaded")
        
        return artifacts
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None

# Load artifacts at startup
print("\n" + "="*50)
print("Loading Models and Artifacts...")
print("="*50)
artifacts = load_artifacts()

if artifacts is None:
    print("\n⚠️ Warning: Models not found. Please run the notebook first to train and save models.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_encode(encoder, value, default=0):
    """Safely encode a value using label encoder"""
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return default
    except:
        return default

def prepare_features(form_data):
    """Prepare features from form data for prediction
    
    Returns separate feature vectors for classification (NO price) 
    and regression (WITH price) to avoid data leakage.
    """
    
    le = artifacts['label_encoders']
    
    # Extract values from form
    bhk = int(form_data.get('bhk', 2))
    size_sqft = float(form_data.get('size_sqft', 1500))
    price = float(form_data.get('price', 50))
    year_built = int(form_data.get('year_built', 2015))
    floor_no = int(form_data.get('floor_no', 5))
    total_floors = int(form_data.get('total_floors', 10))
    nearby_schools = int(form_data.get('nearby_schools', 5))
    nearby_hospitals = int(form_data.get('nearby_hospitals', 3))
    
    state = form_data.get('state', 'Maharashtra')
    city = form_data.get('city', 'Mumbai')
    property_type = form_data.get('property_type', 'Apartment')
    furnished_status = form_data.get('furnished_status', 'Semi-furnished')
    owner_type = form_data.get('owner_type', 'Owner')
    availability = form_data.get('availability', 'Ready_to_Move')
    transport = form_data.get('transport', 'Medium')
    parking = form_data.get('parking', 'Yes')
    security = form_data.get('security', 'Yes')
    
    # Calculate derived features
    current_year = 2025
    age_of_property = current_year - year_built
    
    # Transport score
    transport_map = {'Low': 1, 'Medium': 2, 'High': 3}
    transport_score = transport_map.get(transport, 2)
    
    # Binary features
    is_ready_to_move = 1 if availability == 'Ready_to_Move' else 0
    has_parking = 1 if parking == 'Yes' else 0
    has_security = 1 if security == 'Yes' else 0
    
    # Infrastructure score
    school_density = nearby_schools / 10
    hospital_score = nearby_hospitals / 10
    security_score = has_security
    infrastructure_score = (school_density * 0.25 + hospital_score * 0.25 + 
                           (transport_score / 3) * 0.25 + security_score * 0.25)
    
    # Location premium (simplified)
    location_premium = 0.5  # Default
    
    # Amenities count (simplified)
    amenities_count = 3
    
    # Encode categorical variables
    state_encoded = safe_encode(le.get('State', None), state) if 'State' in le else 0
    city_encoded = safe_encode(le.get('City', None), city) if 'City' in le else 0
    property_type_encoded = safe_encode(le.get('Property_Type', None), property_type) if 'Property_Type' in le else 0
    furnished_encoded = safe_encode(le.get('Furnished_Status', None), furnished_status) if 'Furnished_Status' in le else 0
    owner_encoded = safe_encode(le.get('Owner_Type', None), owner_type) if 'Owner_Type' in le else 0
    
    # Create ALL features dict (includes Price)
    all_features = {
        'BHK': bhk,
        'Size_in_SqFt': size_sqft,
        'Price_in_Lakhs': price,
        'Year_Built': year_built,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age_of_property,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Transport_Score': transport_score,
        'Infrastructure_Score': infrastructure_score,
        'Is_Ready_to_Move': is_ready_to_move,
        'Has_Parking': has_parking,
        'Has_Security': has_security,
        'Amenities_Count': amenities_count,
        'Location_Premium_Score': location_premium,
        'State_Encoded': state_encoded,
        'City_Encoded': city_encoded,
        'Property_Type_Encoded': property_type_encoded,
        'Furnished_Status_Encoded': furnished_encoded,
        'Owner_Type_Encoded': owner_encoded
    }
    
    # Create CLASSIFICATION feature vector (NO PRICE - avoids data leakage)
    clf_feature_names = artifacts['clf_feature_names']
    clf_vector = []
    for feat in clf_feature_names:
        clf_vector.append(all_features.get(feat, 0))
    
    # Create REGRESSION feature vector (WITH PRICE)
    reg_feature_names = artifacts['reg_feature_names']
    reg_vector = []
    for feat in reg_feature_names:
        reg_vector.append(all_features.get(feat, 0))
    
    return (np.array(clf_vector).reshape(1, -1), 
            np.array(reg_vector).reshape(1, -1), 
            all_features)

def get_feature_importance():
    """Get feature importance from the models"""
    try:
        clf_model = artifacts['clf_model']
        reg_model = artifacts['reg_model']
        clf_feature_names = artifacts['clf_feature_names']
        reg_feature_names = artifacts['reg_feature_names']
        
        clf_importance = dict(zip(clf_feature_names, clf_model.feature_importances_))
        reg_importance = dict(zip(reg_feature_names, reg_model.feature_importances_))
        
        # Sort by importance
        clf_importance = dict(sorted(clf_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        reg_importance = dict(sorted(reg_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return clf_importance, reg_importance
    except:
        return {}, {}

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    """Home page with property listing and filters"""
    if artifacts is None:
        return render_template('error.html', 
                             message="Models not loaded. Please run the Jupyter notebook first.")
    
    dropdown_values = artifacts['dropdown_values']
    return render_template('index.html', 
                         states=sorted(dropdown_values['states']),
                         cities=sorted(dropdown_values['cities']),
                         property_types=dropdown_values['property_types'],
                         furnished_status=dropdown_values['furnished_status'],
                         owner_types=dropdown_values['owner_types'],
                         availability_status=dropdown_values['availability_status'],
                         transport_options=['Low', 'Medium', 'High'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if artifacts is None:
        return render_template('error.html', 
                             message="Models not loaded. Please run the Jupyter notebook first.")
    
    dropdown_values = artifacts['dropdown_values']
    
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Extract key values for sanity check
            price = float(form_data.get('price', 50))
            size_sqft = float(form_data.get('size_sqft', 1000))
            bhk = int(form_data.get('bhk', 2))
            city = form_data.get('city', 'Mumbai')
            
            # ============================================================
            # SMART INVESTMENT VALIDATION (Rule-based sanity checks)
            # ============================================================
            # These checks catch obviously bad investments that the ML model
            # might miss due to not having city median comparisons
            
            price_per_sqft = (price * 100000) / size_sqft  # Convert lakhs to rupees
            
            # City-wise average price per sq ft benchmarks (approximate market rates)
            city_benchmarks = {
                'Mumbai': 25000, 'New Delhi': 15000, 'Bangalore': 12000, 
                'Chennai': 10000, 'Kolkata': 8000, 'Hyderabad': 10000,
                'Pune': 10000, 'Ahmedabad': 7000, 'Jaipur': 6000,
                'Lucknow': 5000, 'Chandigarh': 8000, 'Gurgaon': 12000,
                'Noida': 10000, 'Thane': 15000, 'Navi Mumbai': 12000
            }
            
            # Default benchmark for unknown cities
            city_benchmark = city_benchmarks.get(city, 8000)
            
            # Calculate how overpriced/underpriced the property is
            price_ratio = price_per_sqft / city_benchmark
            
            # Rule-based flags
            is_extremely_overpriced = price_ratio > 2.5  # More than 2.5x market rate
            is_overpriced = price_ratio > 1.5  # More than 1.5x market rate
            is_underpriced = price_ratio < 0.7  # Less than 70% of market rate
            is_tiny_property = size_sqft < 200  # Very small property
            is_tiny_expensive = size_sqft < 300 and price > 50  # Small but expensive
            
            # Size-BHK mismatch check
            min_size_per_bhk = {1: 300, 2: 600, 3: 900, 4: 1200, 5: 1500}
            expected_min_size = min_size_per_bhk.get(bhk, 300)
            is_size_mismatch = size_sqft < expected_min_size * 0.5
            
            # Prepare features (separate vectors for classification and regression)
            clf_features, reg_features, feature_dict = prepare_features(form_data)
            
            # Make predictions
            clf_model = artifacts['clf_model']
            reg_model = artifacts['reg_model']
            
            # Classification prediction (uses features WITHOUT price)
            clf_pred = clf_model.predict(clf_features)[0]
            clf_proba = clf_model.predict_proba(clf_features)[0]
            
            # Regression prediction (uses features WITH price)
            reg_pred = reg_model.predict(reg_features)[0]
            
            # ============================================================
            # APPLY RULE-BASED ADJUSTMENTS
            # ============================================================
            warnings = []
            adjusted_confidence_good = clf_proba[1]
            adjusted_confidence_bad = clf_proba[0]
            
            # Penalty for extremely overpriced properties
            if is_extremely_overpriced:
                adjusted_confidence_good *= 0.2  # Reduce to 20%
                adjusted_confidence_bad = 1 - adjusted_confidence_good
                warnings.append(f"⚠️ Extremely overpriced: ₹{price_per_sqft:,.0f}/sqft vs market avg ₹{city_benchmark:,.0f}/sqft ({price_ratio:.1f}x)")
                clf_pred = 0  # Override to NOT good investment
            elif is_overpriced:
                adjusted_confidence_good *= 0.5  # Reduce by 50%
                adjusted_confidence_bad = 1 - adjusted_confidence_good
                warnings.append(f"⚠️ Overpriced: ₹{price_per_sqft:,.0f}/sqft vs market avg ₹{city_benchmark:,.0f}/sqft ({price_ratio:.1f}x)")
            
            # Penalty for tiny properties
            if is_tiny_property:
                adjusted_confidence_good *= 0.6
                adjusted_confidence_bad = 1 - adjusted_confidence_good
                warnings.append(f"⚠️ Very small property: {size_sqft} sq ft may have limited resale value")
            
            # Penalty for size-BHK mismatch
            if is_size_mismatch:
                adjusted_confidence_good *= 0.5
                adjusted_confidence_bad = 1 - adjusted_confidence_good
                warnings.append(f"⚠️ Size mismatch: {bhk} BHK typically needs at least {expected_min_size} sq ft")
            
            # Bonus for underpriced properties
            if is_underpriced and not is_tiny_property:
                adjusted_confidence_good = min(0.95, adjusted_confidence_good * 1.3)
                adjusted_confidence_bad = 1 - adjusted_confidence_good
                warnings.append(f"✅ Good value: ₹{price_per_sqft:,.0f}/sqft is below market avg ₹{city_benchmark:,.0f}/sqft")
            
            # Determine final prediction based on adjusted confidence
            # Simple rule: if good probability > 50%, it's a good investment
            if adjusted_confidence_good >= 0.5:
                clf_pred = 1  # Good Investment
            else:
                clf_pred = 0  # Not Recommended
            
            # Calculate appreciation
            current_price = price
            appreciation = ((reg_pred - current_price) / current_price) * 100
            
            # Get feature importance
            clf_importance, reg_importance = get_feature_importance()
            
            # Prepare results
            results = {
                'good_investment': 'Yes' if clf_pred == 1 else 'No',
                'investment_confidence': round(max(adjusted_confidence_good, adjusted_confidence_bad) * 100, 2),
                'confidence_good': round(adjusted_confidence_good * 100, 2),
                'confidence_bad': round(adjusted_confidence_bad * 100, 2),
                'future_price': round(reg_pred, 2),
                'current_price': current_price,
                'appreciation': round(appreciation, 2),
                'clf_importance': clf_importance,
                'reg_importance': reg_importance,
                'warnings': warnings,
                'price_per_sqft': round(price_per_sqft, 0),
                'city_benchmark': city_benchmark,
                'price_ratio': round(price_ratio, 2),
                'input_features': feature_dict
            }
            
            return render_template('result.html', 
                                 results=results,
                                 form_data=form_data)
            
        except Exception as e:
            return render_template('error.html', message=f"Prediction error: {str(e)}")
    
    return render_template('predict.html',
                         states=sorted(dropdown_values['states']),
                         cities=sorted(dropdown_values['cities']),
                         property_types=dropdown_values['property_types'],
                         furnished_status=dropdown_values['furnished_status'],
                         owner_types=dropdown_values['owner_types'],
                         availability_status=dropdown_values['availability_status'],
                         transport_options=['Low', 'Medium', 'High'])

@app.route('/insights')
def insights():
    """Data insights page"""
    if artifacts is None:
        return render_template('error.html', 
                             message="Models not loaded. Please run the Jupyter notebook first.")
    
    try:
        sample_data = artifacts['sample_data']
        
        # Calculate insights
        insights_data = {
            'total_properties': len(sample_data),
            'avg_price': round(sample_data['Price_in_Lakhs'].mean(), 2),
            'price_range': f"{sample_data['Price_in_Lakhs'].min():.2f} - {sample_data['Price_in_Lakhs'].max():.2f}",
            'avg_size': round(sample_data['Size_in_SqFt'].mean(), 0),
            'cities': sample_data['City'].nunique(),
            'property_types': sample_data['Property_Type'].value_counts().to_dict()
        }
        
        # City-wise average prices
        city_prices = sample_data.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
        insights_data['city_prices'] = city_prices.to_dict()
        
        # Property type prices
        type_prices = sample_data.groupby('Property_Type')['Price_in_Lakhs'].mean()
        insights_data['type_prices'] = type_prices.to_dict()
        
        # Get feature importance
        clf_importance, reg_importance = get_feature_importance()
        insights_data['clf_importance'] = clf_importance
        insights_data['reg_importance'] = reg_importance
        
        return render_template('insights.html', insights=insights_data)
        
    except Exception as e:
        return render_template('error.html', message=f"Error loading insights: {str(e)}")


@app.route('/api/insights')
def api_insights():
    """API endpoint for real-time insights data"""
    if artifacts is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        sample_data = artifacts['sample_data']
        
        # Basic stats
        insights_data = {
            'total_properties': len(sample_data),
            'avg_price': round(sample_data['Price_in_Lakhs'].mean(), 2),
            'avg_size': round(sample_data['Size_in_SqFt'].mean(), 0),
            'cities': sample_data['City'].nunique(),
            'min_price': round(sample_data['Price_in_Lakhs'].min(), 2),
            'max_price': round(sample_data['Price_in_Lakhs'].max(), 2)
        }
        
        # City-wise prices
        city_prices = sample_data.groupby('City')['Price_in_Lakhs'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        insights_data['city_data'] = {
            city: {'avg_price': round(row['mean'], 2), 'count': int(row['count'])}
            for city, row in city_prices.iterrows()
        }
        
        # Property type distribution
        type_dist = sample_data['Property_Type'].value_counts()
        insights_data['property_types'] = type_dist.to_dict()
        
        # Type-wise prices
        type_prices = sample_data.groupby('Property_Type')['Price_in_Lakhs'].mean()
        insights_data['type_prices'] = {k: round(v, 2) for k, v in type_prices.to_dict().items()}
        
        # BHK distribution
        bhk_dist = sample_data['BHK'].value_counts().sort_index()
        insights_data['bhk_distribution'] = bhk_dist.to_dict()
        
        # State-wise data
        state_data = sample_data.groupby('State')['Price_in_Lakhs'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        insights_data['state_data'] = {
            state: {'avg_price': round(row['mean'], 2), 'count': int(row['count'])}
            for state, row in state_data.iterrows()
        }
        
        return jsonify(insights_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/analytics/city/<city>')
def api_city_analytics(city):
    """API endpoint for city-specific analytics"""
    if artifacts is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        sample_data = artifacts['sample_data']
        city_data = sample_data[sample_data['City'] == city]
        
        if len(city_data) == 0:
            return jsonify({'error': 'City not found'}), 404
        
        analytics = {
            'city': city,
            'total_properties': len(city_data),
            'avg_price': round(city_data['Price_in_Lakhs'].mean(), 2),
            'min_price': round(city_data['Price_in_Lakhs'].min(), 2),
            'max_price': round(city_data['Price_in_Lakhs'].max(), 2),
            'avg_size': round(city_data['Size_in_SqFt'].mean(), 0),
            'property_types': city_data['Property_Type'].value_counts().to_dict(),
            'bhk_distribution': city_data['BHK'].value_counts().sort_index().to_dict(),
            'avg_price_per_sqft': round(city_data['Price_in_Lakhs'].sum() * 100000 / city_data['Size_in_SqFt'].sum(), 2)
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/analytics/compare')
def api_compare_cities():
    """API endpoint to compare multiple cities"""
    if artifacts is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        cities = request.args.get('cities', '').split(',')
        if not cities or cities == ['']:
            return jsonify({'error': 'No cities specified'}), 400
        
        sample_data = artifacts['sample_data']
        comparison = {}
        
        for city in cities[:5]:  # Limit to 5 cities
            city = city.strip()
            city_data = sample_data[sample_data['City'] == city]
            
            if len(city_data) > 0:
                comparison[city] = {
                    'avg_price': round(city_data['Price_in_Lakhs'].mean(), 2),
                    'count': len(city_data),
                    'avg_size': round(city_data['Size_in_SqFt'].mean(), 0),
                    'growth_rate': round(artifacts['city_growth_rates'].get(city, 0.08) * 100, 2)
                }
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if artifacts is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        features, feature_dict = prepare_features(data)
        
        clf_model = artifacts['clf_model']
        reg_model = artifacts['reg_model']
        
        clf_pred = clf_model.predict(features)[0]
        clf_proba = clf_model.predict_proba(features)[0]
        reg_pred = reg_model.predict(features)[0]
        
        current_price = float(data.get('price', 50))
        appreciation = ((reg_pred - current_price) / current_price) * 100
        
        return jsonify({
            'good_investment': bool(clf_pred),
            'investment_confidence': round(max(clf_proba) * 100, 2),
            'future_price_5y': round(reg_pred, 2),
            'appreciation_percent': round(appreciation, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/cities/<state>')
def get_cities(state):
    """Get cities for a given state"""
    try:
        city_state_map = artifacts['city_state_map']
        cities = [city for city, st in city_state_map.items() if st == state]
        return jsonify(sorted(cities))
    except:
        return jsonify([])

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Real Estate Investment Advisor")
    print("="*50)
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
