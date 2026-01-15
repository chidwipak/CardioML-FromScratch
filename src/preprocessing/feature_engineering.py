"""
Feature Engineering Module for CVD Prediction
Cardiovascular Disease Prediction System

Create domain-specific features for cardiovascular disease prediction.
"""

import numpy as np
import pandas as pd


def calculate_bmi(height_cm, weight_kg):
    """
    Calculate Body Mass Index.
    
    Formula: BMI = weight(kg) / (height(m))^2
    
    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
    
    Returns:
        BMI values
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return bmi


def categorize_bmi(bmi):
    """
    Categorize BMI into standard WHO categories.
    
    Categories:
        0: Underweight (< 18.5)
        1: Normal (18.5 - 24.9)
        2: Overweight (25.0 - 29.9)
        3: Obese (>= 30.0)
    
    Args:
        bmi: BMI values
    
    Returns:
        BMI categories
    """
    bmi = np.array(bmi)
    categories = np.zeros_like(bmi, dtype=int)
    
    categories[bmi < 18.5] = 0  # Underweight
    categories[(bmi >= 18.5) & (bmi < 25)] = 1  # Normal
    categories[(bmi >= 25) & (bmi < 30)] = 2  # Overweight
    categories[bmi >= 30] = 3  # Obese
    
    return categories


def calculate_pulse_pressure(systolic_bp, diastolic_bp):
    """
    Calculate pulse pressure (difference between systolic and diastolic BP).
    
    Pulse Pressure = Systolic BP - Diastolic BP
    High pulse pressure (>60) indicates arterial stiffness.
    
    Args:
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
    
    Returns:
        Pulse pressure values
    """
    return systolic_bp - diastolic_bp


def calculate_mean_arterial_pressure(systolic_bp, diastolic_bp):
    """
    Calculate Mean Arterial Pressure.
    
    Formula: MAP = Diastolic BP + (Pulse Pressure / 3)
    Or: MAP = (2 * Diastolic + Systolic) / 3
    
    Args:
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
    
    Returns:
        MAP values
    """
    return (2 * diastolic_bp + systolic_bp) / 3.0


def categorize_blood_pressure(systolic_bp, diastolic_bp):
    """
    Categorize blood pressure according to AHA guidelines.
    
    Categories:
        0: Normal (< 120/80)
        1: Elevated (120-129/<80)
        2: Hypertension Stage 1 (130-139/80-89)
        3: Hypertension Stage 2 (>=140/>=90)
        4: Hypertensive Crisis (>180/>120)
    
    Args:
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
    
    Returns:
        BP categories
    """
    systolic_bp = np.array(systolic_bp)
    diastolic_bp = np.array(diastolic_bp)
    
    categories = np.zeros(len(systolic_bp), dtype=int)
    
    # Hypertensive Crisis
    mask = (systolic_bp > 180) | (diastolic_bp > 120)
    categories[mask] = 4
    
    # Hypertension Stage 2
    mask = (systolic_bp >= 140) | (diastolic_bp >= 90)
    mask = mask & (categories == 0)
    categories[mask] = 3
    
    # Hypertension Stage 1
    mask = ((systolic_bp >= 130) & (systolic_bp < 140)) | ((diastolic_bp >= 80) & (diastolic_bp < 90))
    mask = mask & (categories == 0)
    categories[mask] = 2
    
    # Elevated
    mask = (systolic_bp >= 120) & (systolic_bp < 130) & (diastolic_bp < 80)
    mask = mask & (categories == 0)
    categories[mask] = 1
    
    # Normal: already set to 0
    
    return categories


def extract_age_features(age_days):
    """
    Extract multiple age-related features.
    
    Args:
        age_days: Age in days
    
    Returns:
        dict: Dictionary of age features
    """
    age_years = age_days / 365.25
    
    return {
        'age_years': age_years,
        'age_decades': age_years / 10.0,
        'age_squared': age_years ** 2,
        'age_group': np.digitize(age_years, bins=[0, 40, 50, 60, 70, 100])
    }


def calculate_metabolic_risk_score(cholesterol, glucose, bmi):
    """
    Calculate composite metabolic risk score.
    
    Combines cholesterol, glucose, and BMI into a single risk metric.
    Higher values indicate higher metabolic risk.
    
    Args:
        cholesterol: Cholesterol level (1=normal, 2=above, 3=well above)
        glucose: Glucose level (1=normal, 2=above, 3=well above)
        bmi: Body Mass Index
    
    Returns:
        Metabolic risk scores
    """
    cholesterol = np.array(cholesterol)
    glucose = np.array(glucose)
    bmi = np.array(bmi)
    
    # Normalize BMI to 0-3 scale (similar to cholesterol/glucose)
    bmi_score = np.zeros_like(bmi)
    bmi_score[bmi < 18.5] = 0.5  # Underweight
    bmi_score[(bmi >= 18.5) & (bmi < 25)] = 1  # Normal
    bmi_score[(bmi >= 25) & (bmi < 30)] = 2  # Overweight
    bmi_score[bmi >= 30] = 3  # Obese
    
    # Combined risk score
    risk_score = (cholesterol - 1) + (glucose - 1) + (bmi_score - 1)
    
    return risk_score


def calculate_lifestyle_risk_score(smoke, alco, active):
    """
    Calculate lifestyle risk score.
    
    Combines smoking, alcohol consumption, and physical activity.
    Higher values indicate riskier lifestyle.
    
    Args:
        smoke: Smoking status (0=no, 1=yes)
        alco: Alcohol consumption (0=no, 1=yes)
        active: Physical activity (0=no, 1=yes)
    
    Returns:
        Lifestyle risk scores
    """
    smoke = np.array(smoke)
    alco = np.array(alco)
    active = np.array(active)
    
    # Risk increases with smoking and alcohol, decreases with activity
    risk_score = smoke + alco - active
    
    return risk_score


def create_interaction_features(age_years, bmi, systolic_bp, diastolic_bp):
    """
    Create interaction features between important variables.
    
    Args:
        age_years: Age in years
        bmi: Body Mass Index
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
    
    Returns:
        dict: Dictionary of interaction features
    """
    return {
        'age_bmi_interaction': age_years * bmi,
        'age_bp_interaction': age_years * (systolic_bp + diastolic_bp) / 2
    }


def engineer_all_features(df):
    """
    Apply all feature engineering transformations to the dataset.
    
    Args:
        df: DataFrame with original features
    
    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()
    
    # BMI features
    df['bmi'] = calculate_bmi(df['height'], df['weight'])
    df['bmi_category'] = categorize_bmi(df['bmi'])
    
    # Blood pressure features
    df['pulse_pressure'] = calculate_pulse_pressure(df['ap_hi'], df['ap_lo'])
    df['mean_arterial_pressure'] = calculate_mean_arterial_pressure(df['ap_hi'], df['ap_lo'])
    df['bp_category'] = categorize_blood_pressure(df['ap_hi'], df['ap_lo'])
    
    # Age features
    age_features = extract_age_features(df['age'])
    for key, value in age_features.items():
        df[key] = value
    
    # Risk scores
    df['metabolic_risk_score'] = calculate_metabolic_risk_score(
        df['cholesterol'], df['gluc'], df['bmi']
    )
    df['lifestyle_risk_score'] = calculate_lifestyle_risk_score(
        df['smoke'], df['alco'], df['active']
    )
    
    # Interaction features
    interaction_features = create_interaction_features(
        df['age_years'], df['bmi'], df['ap_hi'], df['ap_lo']
    )
    for key, value in interaction_features.items():
        df[key] = value
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering Module...")
    
    # Create sample CVD data
    np.random.seed(42)
    n_samples = 5
    
    sample_data = pd.DataFrame({
        'age': np.random.randint(15000, 23000, n_samples),
        'height': np.random.randint(150, 185, n_samples),
        'weight': np.random.uniform(50, 100, n_samples),
        'ap_hi': np.random.randint(110, 160, n_samples),
        'ap_lo': np.random.randint(70, 100, n_samples),
        'cholesterol': np.random.randint(1, 4, n_samples),
        'gluc': np.random.randint(1, 4, n_samples),
        'smoke': np.random.randint(0, 2, n_samples),
        'alco': np.random.randint(0, 2, n_samples),
        'active': np.random.randint(0, 2, n_samples)
    })
    
    print("\nOriginal features:")
    print(sample_data)
    
    print("\n" + "="*60)
    engineered_df = engineer_all_features(sample_data)
    
    print("\nEngineered features:")
    new_features = [col for col in engineered_df.columns if col not in sample_data.columns]
    print(engineered_df[new_features])
    
    print(f"\n✓ Added {len(new_features)} new features!")
    print(f"  Features: {', '.join(new_features)}")
